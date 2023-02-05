import math
import discoart
import numpy
import numpy as np
import torch
import cv2
import torchvision
from PIL import Image, ImageEnhance
from torchvision import transforms
from docarray import Document
import h5py
import glob
import random

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


class generatePipelinee:

    def __init__(self):
        # Constants
        self.prompt = "A galaxy cluster"
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Unet(
            dim=64,
            dim_mults=(1, 2, 4, 8)
        ).to(device=self.DEVICE)

        self.diffusion = GaussianDiffusion(
            self.model,
            timesteps=1000,
            loss_type='l1'
        ).to(device=self.DEVICE)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        self.transform2 = transforms.Compose(
            transforms.Resize((256, 256)),
        )

        self.trainer = Trainer(
            self.diffusion,
            './data/probes/',
            logdir='./logs/probes/',
            image_size=256,
            train_batch_size=16,
            train_lr=2e-5,
            train_num_steps=750001,  # total training steps
            gradient_accumulate_every=2,  # gradient accumulation steps
            ema_decay=0.995,  # exponential moving average decay
            num_workers=32,
            rank=[0]
        )

        self.trainer.load("./weights/probes_model_00745000.pt")

        self.classifier = torchvision.models.resnet18(pretrained=True).to("cuda")
        num_ftrs = self.classifier.fc.in_features
        self.classifier.fc = torch.nn.Linear(num_ftrs, 10)
        self.classifier.fc.to("cuda")

        self.indexOptions = ['Disturbed', 'Merging', 'Round Smooth', 'In-between Round Smooth', 'Cigar Round Smooth'
            , 'Barred Spiral', 'Unbarred Tight Spiral', 'Unbarred Loose Spiral', 'Edge-on without Bulge'
            , 'Edge-on with Bulge']

        self.classifier.load_state_dict(torch.load("./weights/classifier.pt"))
        self.classifier.eval()
        self.classifier.to("cuda")

        self.sine_importance_map = np.zeros((256,256,3))

        for i in range(256):
            for j in range(256):
                val_i = math.sin((i/256)*math.pi)
                val_j = math.sin((i/256)*math.pi)
                out = val_i * val_j

                self.sine_importance_map[i,j,0] = out
                self.sine_importance_map[i,j,1] = out
                self.sine_importance_map[i,j,2] = out

    def setInput(self, img):
        img = torch.from_numpy(img)
        img_tensor = self.transform(img)
        self.diffusion.inputImg = img_tensor.to("cuda")

    def gamma_mapping(self, img, gamma):
        re_img = img ** (1 / gamma)
        return re_img

    # this gives rating for whether the image is a good galaxy
    def rating(self, img):
        nonzero = torch.count_nonzero(img)
        pix = img.shape[0] * img.shape[1] * img.shape[2]

        sum = torch.sum(img)
        mean = sum / pix

        return nonzero / pix, mean

    def upscale(self, img):
        upscaled_image = self.upscaler(prompt=self.prompt, image=img)
        return upscaled_image

    def pad_3d_tensor(self,tensor, pad_size):
        # Get the original shape of the tensor
        original_shape = tensor.shape
        tensor_height, tensor_width = original_shape[-2:]

        # Create zero tensors to add as padding
        left_right_pad = torch.zeros((original_shape[0], tensor_height, pad_size), dtype=tensor.dtype)
        top_bottom_pad = torch.zeros((original_shape[0], pad_size, tensor_width + 2 * pad_size), dtype=tensor.dtype)

        # Concatenate the tensor with the padding on left and right sides
        padded_tensor = torch.cat((left_right_pad, tensor, left_right_pad), dim=-1)

        # Concatenate the padded tensor with the padding on top and bottom
        padded_tensor = torch.cat((top_bottom_pad, padded_tensor, top_bottom_pad), dim=-2)

        return padded_tensor

    def shape_filter(self, img):
        return numpy.multiply(img, self.sine_importance_map)

    def generate(self, inputImg: torch.Tensor = None):
        if inputImg != None:
            inputImg = self.transform(inputImg)
            inputImg = inputImg[0:3, :, :]

            #pad the image on each side with 50 pixels of black
            inputImg = self.pad_3d_tensor(inputImg,15)


            #crop the image to 256x256
            inputImg = torch.nn.functional.interpolate(input=inputImg.unsqueeze(0),
                                                       size=(256, 256), mode='bilinear')

            inputImg = inputImg.squeeze(0)

            inputImg = inputImg.to("cuda")
        self.diffusion.inputImg = inputImg
        sampled_batch = self.diffusion.sample(256, batch_size=1)
        i = 0
        raws = []
        outputs = []

        sample = sampled_batch[0]
        sample = (sample - sample.min()) / (sample.max() - sample.min())
        print(self.rating(sample))

        while self.rating(sample)[0] < 0.15 \
                or (self.rating(sample)[0] > 0.9 and self.rating(sample)[1] > 0.04):
            self.diffusion.inputImg = None
            sampled_batch = self.diffusion.sample(256, batch_size=1)
            sample = sampled_batch[0]
            sample = (sample - sample.min()) / (sample.max() - sample.min())
            print(self.rating(sample))

        raws.append(sample)

        t_sample = self.gamma_mapping(sample, 2.5)
        # sample = self.gamma_mapping(sample, 2)
        # raws.append(sample)
        classOut = self.classifier(raws[0].unsqueeze(0).to("cuda"))
        classOut = torch.nn.functional.softmax(classOut, dim=1)
        classOut = classOut.cpu().detach().numpy()
        maxIndex = np.argmax(classOut)
        print(self.indexOptions[maxIndex], classOut[0][maxIndex])

        raw = raws[0]
        raw = raw.cpu().numpy()
        raw = numpy.transpose(raw, (1, 2, 0))
        raw = self.shape_filter(raw)
        raw = raw * 255
        raw = raw.astype(np.uint8)

        cv2.imwrite("rarRuntime.png",raw)

        return raw, self.indexOptions[maxIndex]


if __name__ == "__main__":
    pipeline = generatePipelinee()
    img, string = pipeline.generate()
