import argparse
import sys
sys.path.append('./taming-transformers')
from IPython import display
from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm
from CLIP import clip
import numpy as np
import imageio
from PIL import ImageFile, Image
from imgtag import ImgTag    # metadatos
from libxmp import *         # metadatos
import libxmp                # metadatos
from stegano import lsb
import json
from datetime import date
import os
from ReplaceGrad import ReplaceGrad
from ClampWithGrad import ClampWithGrad
from Prompt import Prompt
from MakeCutouts import MakeCutouts

class VQGAN:
    def __init__(self, input_text, width, height, model, save_increment,initial_image, final_image, seed, max_iterations, save_dir, input_images,model_dir="/content"):
        self.text=input_text
        self.width=width
        self.height=height
        self.model=model
        self.image_interval=save_increment
        self.initial_image=initial_image
        self.images_objective=final_image
        self.seed=seed
        self.max_iterations=max_iterations
        self.save_dir=save_dir
        self.input_images=input_images
        self.model_dir=model_dir
        self.parse_args()


    def parse_args(self):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        if self.seed == -1:
            self.seed = None
        if self.initial_image == "None":
            self.initial_image = None
        if self.images_objective == "None" or not self.images_objective:
            self.images_objective = []
        else:
            self.images_objective = self.image_objective.split("|")
            self.images_objective = [image.strip() for image in self.images_objective]
        if self.initial_image or self.images_objective != []:
            self.input_images = True
        self.input_text=self.text
        self.text = [phrase.strip() for phrase in self.text.split("|")]
        if self.text == ['']:
            self.text = []
        self.vqgan_config=os.path.join(self.model_dir,f'{self.model}.yaml')
        print(f"Using {self.vqgan_config} as model")
        self.args = argparse.Namespace(
            prompts=self.text,
            image_prompts=self.images_objective,
            noise_prompt_seeds=[],
            noise_prompt_weights=[],
            size=[self.width, self.height],
            init_image=self.initial_image,
            init_weight=0.,
            clip_model='ViT-B/32',
            vqgan_config=self.vqgan_config,
            vqgan_checkpoint=f'{self.model}.ckpt',
            step_size=0.1,
            cutn=64,
            cut_pow=1.,
            display_freq=self.image_interval,
            seed=self.seed,
        )

    def run(self):
        self.replace_grad = ReplaceGrad.apply
        self.clamp_with_grad = ClampWithGrad.apply
        print("Running.")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        if self.text:
            print('Using texts:', self.text)
        if self.images_objective:
            print('Using image prompts:', self.images_objective)
        if self.args.seed is None:
            seed = torch.seed()
        else:
            seed = self.args.seed
        torch.manual_seed(seed)
        print('Using seed:', seed)
        self.vqmodel = self.load_vqgan_model(self.args.vqgan_config, self.args.vqgan_checkpoint).to(device)
        self.perceptor = clip.load(self.args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)
        cut_size = self.perceptor.visual.input_resolution
        e_dim = self.vqmodel.quantize.e_dim
        f = 2 ** (self.vqmodel.decoder.num_resolutions - 1)
        self.make_cutouts = MakeCutouts(cut_size, self.args.cutn, cut_pow=self.args.cut_pow)
        n_toks = self.vqmodel.quantize.n_e
        toksX, toksY = self.args.size[0] // f, self.args.size[1] // f
        sideX, sideY = toksX * f, toksY * f
        self.z_min = self.vqmodel.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        self.z_max = self.vqmodel.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
        if self.args.init_image:
            pil_image = Image.open(self.args.init_image).convert('RGB')
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            self.z, *_ = self.vqmodel.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
        else:
            one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
            self.z = one_hot @ self.vqmodel.quantize.embedding.weight
            self.z = self.z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
        self.z_orig = self.z.clone()
        self.z.requires_grad_(True)
        self.opt = optim.Adam([self.z], lr=self.args.step_size)
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711])
        self.pMs = []
        for prompt in self.args.prompts:
            txt, weight, stop = self.parse_prompt(prompt)
            embed = self.perceptor.encode_text(clip.tokenize(txt).to(device)).float()
            self.pMs.append(Prompt(embed, weight, stop).to(device))

        for prompt in self.args.image_prompts:
            path, weight, stop = self.parse_prompt(prompt)
            img = self.resize_image(Image.open(path).convert('RGB'), (sideX, sideY))
            batch = self.make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
            embed = self.perceptor.encode_image(self.normalize(batch)).float()
            self.pMs.append(Prompt(embed, weight, stop).to(device))

        for seed, weight in zip(self.args.noise_prompt_seeds, self.args.noise_prompt_weights):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, self.perceptor.visual.output_dim]).normal_(generator=gen)
            self.pMs.append(Prompt(embed, weight).to(device))

        self.i = 0
        try:
            with tqdm() as pbar:
                while True:
                    self.train(self.i)
                    if self.i == self.max_iterations:
                        break
                    self.i += 1
                    pbar.update()
        except KeyboardInterrupt:
            pass

    def vector_quantize(self,x, codebook):
        d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
        indices = d.argmin(-1)
        x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
        return self.replace_grad(x_q, x)

    def parse_prompt(self,prompt):
        vals = prompt.rsplit(':', 2)
        vals = vals + ['', '1', '-inf'][len(vals):]
        return vals[0], float(vals[1]), float(vals[2])

    def load_vqgan_model(self,config_path, checkpoint_path):
        config = OmegaConf.load(config_path)
        if config.model.target == 'taming.models.vqgan.VQModel':
            self.vqmodel = vqgan.VQModel(**config.model.params)
            self.vqmodel.eval().requires_grad_(False)
            self.vqmodel.init_from_ckpt(checkpoint_path)
        elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
            parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
            parent_model.eval().requires_grad_(False)
            parent_model.init_from_ckpt(checkpoint_path)
            self.vqmodel = parent_model.first_stage_model
        else:
            raise ValueError(f'unknown model type: {config.model.target}')
        del self.vqmodel.loss
        return self.vqmodel

    def resize_image(image, out_size):
        ratio = image.size[0] / image.size[1]
        area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
        size = round((area * ratio) ** 0.5), round((area / ratio) ** 0.5)
        return image.resize(size, Image.LANCZOS)

    def save_or_createdir(self,img, save_dir, input_text):
        today = date.today()
        todaystr = today.strftime("%b-%d-%Y") + "/"
        if os.path.isdir(save_dir + todaystr) == False:
            os.mkdir(save_dir + todaystr)
        input_txt_str = input_text + "/"
        if os.path.isdir(save_dir + todaystr + input_txt_str) == False:
            os.mkdir(save_dir + todaystr + input_txt_str)
        index_flag = True
        index = 0
        while index_flag:
            im_fnam = save_dir + todaystr + input_txt_str + str(index) + ".png"
            if os.path.exists(im_fnam):
                index += 1
            else:
                imageio.imwrite(im_fnam, np.array(img))
                index_flag = False
                print("Saving image " + im_fnam)

    def synth(self,z):
        z_q = self.vector_quantize(z.movedim(1, 3), self.vqmodel.quantize.embedding.weight).movedim(3, 1)
        return self.clamp_with_grad(self.vqmodel.decode(z_q).add(1).div(2), 0, 1)

    def add_xmp_data(self,filename):
        images = ImgTag(filename=filename)
        images.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'creator', 'VQGAN+CLIP',
                                     {"prop_array_is_ordered": True, "prop_value_is_array": True})
        if self.args.prompts:
            images.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'title', " | ".join(self.args.prompts),
                                         {"prop_array_is_ordered": True, "prop_value_is_array": True})
        else:
            images.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'title', 'None',
                                         {"prop_array_is_ordered": True, "prop_value_is_array": True})
        images.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'i', str(self.i),
                                     {"prop_array_is_ordered": True, "prop_value_is_array": True})
        images.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'model', self.number_model,
                                     {"prop_array_is_ordered": True, "prop_value_is_array": True})
        images.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'seed', str(self.seed),
                                     {"prop_array_is_ordered": True, "prop_value_is_array": True})
        images.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'input_images', str(self.input_images),
                                     {"prop_array_is_ordered": True, "prop_value_is_array": True})
        # for frases in args.prompts:
        #    imagen.xmp.append_array_item(libxmp.consts.XMP_NS_DC, 'Prompt' ,frases, {"prop_array_is_ordered":True, "prop_value_is_array":True})
        images.close()

    def add_stegano_data(self,filename):
        data = {
            "title": " | ".join(self.args.prompts) if self.args.prompts else None,
            "notebook": "VQGAN+CLIP",
            "i": self.i,
            "model": self.number_model,
            "seed": str(self.seed),
            "input_images": self.input_images
        }
        lsb.hide(filename, json.dumps(data)).save(filename)

    @torch.no_grad()
    def checkin(self,i, losses):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
        out = self.synth(self.z)
        current_img = TF.to_pil_image(out[0].cpu())
        current_img.save('progress.png')
        self.save_or_createdir(current_img, self.save_dir, self.input_text)
        self.add_stegano_data('progress.png')
        self.add_xmp_data('progress.png')
        display.display(display.Image('progress.png'))

    def ascend_txt(self):
        out = self.synth(self.z)
        iii = self.perceptor.encode_image(self.normalize(self.make_cutouts(out))).float()

        result = []

        if self.args.init_weight:
            result.append(F.mse_loss(self.z, self.z_orig) * self.args.init_weight / 2)

        for prompt in self.pMs:
            result.append(prompt(iii))
        img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:, :, :]
        img = np.transpose(img, (1, 2, 0))
        filename = f"steps/{self.i:04}.png"
        imageio.imwrite(filename, np.array(img))
        self.add_stegano_data(filename)
        self.add_xmp_data(filename)
        return result

    def train(self,i):
        self.opt.zero_grad()
        lossAll = self.ascend_txt()
        if self.i % self.args.display_freq == 0:
            self.checkin(i, lossAll)
        loss = sum(lossAll)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))

    def get_number_model(self):
        model=self.model
        number_models = {"vqgan_imagenet_f16_16384": "ImageNet 16384", "vqgan_imagenet_f16_1024": "ImageNet 1024",
                         "wikiart_1024": "WikiArt 1024", "wikiart_16384": "WikiArt 16384", "coco": "COCO-Stuff",
                         "faceshq": "FacesHQ", "sflckr": "S-FLCKR"}
        return number_models[model]

    def set_number_model(self,number_model):
        self.number_model=number_model