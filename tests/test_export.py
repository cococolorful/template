import os
import torch
from torch import optim, nn, utils
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
import numpy as np
import onnxruntime

# define any number of nn.Modules (or use your current ones)
import template
from template.model import LitAutoEncoder

# init the autoencoder
autoencoder = LitAutoEncoder()

# setup data
dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset)

# train the model
trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)

# load checkpoint
checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint)

# Specify the ONNX file path and export model
filepath =  './lightning_logs/version_0/model.onnx'
autoencoder.to_onnx(filepath, export_params=True, input_names=['input'], output_names=['output'], opset_version=18)

# Load the ONNX model using ONNX Runtime
ort_session = onnxruntime.InferenceSession(filepath)

# Prepare input data
input_name = ort_session.get_inputs()[0].name
ort_inputs = {input_name: autoencoder.example_input_array.numpy()}

# Get output from ONNX model
ort_outs = ort_session.run(None, ort_inputs)

# Check the output is the same as the original model
with torch.no_grad():
    model_result_in_torch = autoencoder(autoencoder.example_input_array.to(autoencoder.device)).cpu().numpy()

# Validate the correctness of the ONNX model
is_valid = np.allclose(ort_outs[0], model_result_in_torch, atol=1e-6)

if is_valid:
    print("✅ ONNX model exported successfully and is valid!")
else:
    print("❌ ONNX model export failed: outputs do not match!")

exit()