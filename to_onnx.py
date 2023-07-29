import torch
from models import Wav2Lip

torch_model = Wav2Lip()

torch_model_path = "../checkpoints/wav2lip.pth"
torch_checkpoint = torch.load(torch_model_path)
s = torch_checkpoint["state_dict"]
try:
    torch_model.load_state_dict(s)
except RuntimeError as e:
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    torch_model.load_state_dict(new_s)


BATCH_SIZE = 1
img_input = torch.randn(BATCH_SIZE, 6, 96, 96)
mel_input = torch.randn(BATCH_SIZE, 1, 80, 16)

export_onnx_path = "../checkpoints/wav2lip.onnx"
torch.onnx.export(
    torch_model,
    args=(mel_input, img_input),
    f=export_onnx_path,
    input_names=['mel_input', 'img_input'],
    output_names=['img_output'],
)


# trtexec --onnx=wav2lip.onnx --saveEngine=wav2lip.trt
# trtexec --shapes=img_input:1x6x96x96,mel_input:1x1x80x16 --loadEngine=wav2lip.trt
