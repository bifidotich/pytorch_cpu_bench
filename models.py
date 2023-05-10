import time
import torch
import datetime
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from rudalle.pipelines import generate_images, show, super_resolution, cherry_pick_by_ruclip
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan
from rudalle.utils import seed_everything


def benchmark(name):
    def dec(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            func(*args, **kwargs)
            end = time.time()
            res = f'[bench] {name} - Time is {format(end-start)} sec.'
            print(res)
            return res
        return wrapper
    return dec


class DialoGPT:

    def __init__(self,
                 device='cpu',
                 model_path='ruDialoGPT-medium',
                 num_threads=None):

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        if num_threads: torch.set_num_threads(num_threads)

    def __post_init__(self):
        pass

    @benchmark(name='ruDialoGPT-medium')
    def work(self, text_):

        input_ids = self.tokenizer.encode(text_, return_tensors="pt").to(self.device)
        out = self.model.generate(
            input_ids,
            top_k=10,
            top_p=0.95,
            num_beams=3,
            num_return_sequences=3,
            do_sample=True,
            no_repeat_ngram_size=2,
            temperature=1.2,
            repetition_penalty=1.2,
            length_penalty=1.0,
            eos_token_id=50257,
            max_new_tokens=40
        )
        res = list(map(self.tokenizer.decode, out))[0]
        res_text = (res.split("@@ПЕРВЫЙ@@")[0]).split("@@ВТОРОЙ@@")[-1]
        print(f"{text_} ----> {res_text}")
        return res_text


class Malevich:

    def __init__(self,
                 device='cpu',
                 model_path='rudalle-Malevich',
                 num_threads=None):

        if device == "cpu":
            fp16 = False
        else:
            fp16 = True
        self.device = torch.device(device)
        self.tokenizer = get_tokenizer()
        self.model = get_rudalle_model("Malevich", pretrained=True, fp16=fp16, device=device)
        self.vae = get_vae(dwt=True).to(device)
        if num_threads: torch.set_num_threads(num_threads)

    def __post_init__(self):
        pass

    @benchmark(name='rudalle-Malevich')
    def work(self, text_, seed=5555):

        seed_everything(seed)
        pil_images = []
        scores = []
        for top_k, top_p, images_num in [(3048, 0.995, 3), ]:
            _pil_images, _scores = generate_images(text_, self.tokenizer, self.model, self.vae, top_k=top_k, images_num=images_num,
                                                   bs=8,
                                                   top_p=top_p)
            pil_images += _pil_images
            scores += _scores
            torch.cuda.empty_cache()

        res_path = []
        strdt = str(datetime.datetime.now().strftime("%I_%M_%B_%d_%Y"))
        for i, image in enumerate(pil_images):
            image_name = f"image_{strdt}_{i}.jpg"
            res_path.append(image_name)
            image.save(image_name)

        return res_path
