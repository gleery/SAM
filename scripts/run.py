from argparse import Namespace
import os
import time
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import sys
sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from datasets.augmentations import AgeTransformer
from utils.common import tensor2im, log_image
from options.test_options import TestOptions
from models.psp import pSp

img_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

def load_model(model_path):
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
	# opts.update(vars(test_opts))
	opts = Namespace(**opts)

	net = pSp(opts)
	net.eval()
	net.cuda()
    return net

def run_alignment(image_path):
    import dlib
    from scripts.align_all_parallel import align_face
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor) 
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image 

def run(image_path):
	test_opts = TestOptions().parse()
    target_ages = [20, 30, 40]

    net = load_model(test_opts.checkpoint_path)

	age_transformers = [AgeTransformer(target_age=age) for age in arget_age]

	original_image = Image.open(image_path).convert("RGB")
    #original_image.resize((256, 256))
    aligned_image = run_alignment(image_path)
    aligned_image.resize((256, 256))


    img_transforms = EXPERIMENT_ARGS['transform']
    input_image = img_transforms(aligned_image)

    target_ages = [50]
    age_transformers = [AgeTransformer(target_age=age) for age in target_ages]

    results = np.array(aligned_image.resize((256, 256)))
    for age_transformer in age_transformers:
        print(f"Running on target age: {age_transformer.target_age}")
        with torch.no_grad():
            input_image_age = [age_transformer(input_image.cpu()).to('cuda')]
            print (input_image.size())
            input_image_age = torch.stack(input_image_age)
            result_tensor = run_on_batch(input_image_age, net)[0]
            print (result_tensor.size())
            result_image = tensor2im(result_tensor)
            #results = np.concatenate([results, result_image], axis=1)
            results = np.array(result_image)


def run_on_batch(inputs, net):
    result_batch = net(inputs.to("cuda").float(), randomize_noise=False, resize=False)
    return result_batch


if __name__ == '__main__':
	run()

