import os
from models import DialoGPT, Malevich
from config import ModelsName, TextGen, SeedGen, KfX


def check_model(name):
    if not os.path.isdir(name):
        print(f"{name} - Not found. Please run hf_install.bat")
        return False
    else:
        return True


def cos(func):
    def wrapper():
        print("=" * 100)
        func()
        print("=" * 100)
    return wrapper


@cos
def bench_DialoGPT():
    print(f"bench_DialoGPT - start")
    print(f"1x")
    model = DialoGPT()
    return model.work(text_=TextGen)


@cos
def bench_Malevich():
    print("load in [disc]/tmp/rudalle")
    print(f"bench_Malevich - start")
    print(f"1x")
    model = Malevich()
    return model.work(text_=TextGen, seed=SeedGen)


if __name__ == '__main__':
    results = []
    if False not in [check_model(iter) for iter in ModelsName]:
        results.append(bench_DialoGPT())
        results.append(bench_Malevich())
        for iter in results: print(iter)
        




