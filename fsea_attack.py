import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import timm
import os
import argparse
import tqdm
from PIL import Image

img_height, img_width = 224, 224
img_max, img_min = 1., 0

class Attack:
    """
    Base class for attacks
    """
    def __init__(self, attack, model_name, epsilon, targeted, random_start, norm, loss, device=None):
        if norm not in ['l2', 'linfty']:
            raise Exception(f"Unsupported norm {norm}")
        self.attack = attack
        self.model = self.load_model(model_name)
        self.epsilon = epsilon
        self.targeted = targeted
        self.random_start = random_start
        self.norm = norm
        # Check if model parameters are available
        try:
            self.device = next(self.model.parameters()).device if device is None else device
        except StopIteration:
            raise RuntimeError("Model has no parameters or failed to initialize.")
        self.loss = self.loss_function(loss)

    def load_model(self, model_name):
        def load_single_model(name):
            try:
                if name in models.__dict__.keys():
                    print(f'=> Loading model {name} from torchvision.models')
                    model = models.__dict__[name](weights="DEFAULT")
                elif name in timm.list_models():
                    print(f'=> Loading model {name} from timm.models')
                    model = timm.create_model(name, pretrained=True)
                else:
                    raise ValueError(f'Model {name} not supported')
                model = wrap_model(model.eval())
                # Ensure model is moved to GPU
                if torch.cuda.is_available():
                    model = model.cuda()
                else:
                    raise RuntimeError(f"CUDA is not available for model {name}")
                # Check model parameters
                if not list(model.parameters()):
                    raise RuntimeError(f"Model {name} has no parameters")
                return model
            except Exception as e:
                raise RuntimeError(f"Failed to load model {name}: {str(e)}")

        if isinstance(model_name, list):
            models_list = [load_single_model(name) for name in model_name]
            if not models_list:
                raise RuntimeError("No models loaded for ensemble")
            return EnsembleModel(models_list)
        else:
            return load_single_model(model_name)

    def loss_function(self, loss):
        if loss == 'crossentropy':
            return nn.CrossEntropyLoss()
        else:
            raise Exception(f"Unsupported loss {loss}")

    def get_logits(self, x, **kwargs):
        return self.model(x)

    def get_loss(self, logits, label):
        return -self.loss(logits, label) if self.targeted else self.loss(logits, label)

    def get_grad(self, loss, delta, **kwargs):
        return torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

    def get_momentum(self, grad, momentum, **kwargs):
        return momentum * self.decay + grad / (grad.abs().mean(dim=(1,2,3), keepdim=True))

    def init_delta(self, data, **kwargs):
        delta = torch.zeros_like(data).to(self.device)
        if self.random_start:
            if self.norm == 'linfty':
                delta.uniform_(-self.epsilon, self.epsilon)
            else:
                delta.normal_(-self.epsilon, self.epsilon)
                d_flat = delta.view(delta.size(0), -1)
                n = d_flat.norm(p=2, dim=-1).view(delta.size(0), 1, 1, 1)
                r = torch.zeros_like(data).uniform_(0,1).to(self.device)
                delta *= r/n*self.epsilon
            delta = clamp(delta, img_min-data, img_max-data)
        delta.requires_grad = True
        return delta

    def update_delta(self, delta, data, grad, alpha, **kwargs):
        if self.norm == 'linfty':
            delta = torch.clamp(delta + alpha * grad.sign(), -self.epsilon, self.epsilon)
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
            scaled_grad = grad / (grad_norm + 1e-20)
            delta = (delta + scaled_grad * alpha).view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(delta)
        delta = clamp(delta, img_min-data, img_max-data)
        return delta.detach().requires_grad_(True)

    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        delta = self.init_delta(data)
        momentum = 0
        for _ in range(self.epoch):
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))
            loss = self.get_loss(logits, label)
            grad = self.get_grad(loss, delta)
            momentum = self.get_momentum(grad, momentum)
            delta = self.update_delta(delta, data, momentum, self.alpha)
        return delta.detach()

    def transform(self, data, **kwargs):
        return data

    def __call__(self, *input, **kwargs):
        self.model.eval()
        return self.forward(*input, **kwargs)

class FSEA(Attack):
    """
    FSEA attack class, inherits from Attack base class
    """
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=0, 
                 targeted=False, random_start=True, freq_reg_weight=0.5,
                 norm='linfty', loss='crossentropy', device=None, attack='FSEA', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.freq_reg_weight = freq_reg_weight
        self.num_model = len(model_name) if isinstance(model_name, list) else 1

    def forward(self, data, label, **kwargs):
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        B, C, H, W = data.size()
        loss_func = nn.CrossEntropyLoss()
        momentum_G = 0.
        delta = torch.zeros_like(data).to(self.device) + 0.001 * torch.randn(data.shape, device=self.device)
        delta.requires_grad = True

        for i in range(self.epoch):
            outputs = [self.model.models[idx](data + delta) for idx in range(self.num_model)] if isinstance(self.model, EnsembleModel) else [self.model(data + delta)]
            losses = [loss_func(outputs[idx], label) for idx in range(len(outputs))]
            grads = [torch.autograd.grad(losses[idx], delta, retain_graph=True)[0] for idx in range(len(outputs))]
            total_grad = self.frequency_guidance(grads, delta, H, W)
            momentum_G = self.get_momentum(total_grad, momentum_G)
            delta = self.update_delta(delta, data, momentum_G, self.alpha)
            avg_loss = sum(losses) / len(losses)
            print(f"Epoch {i+1}/{self.epoch}: Avg CE Loss = {avg_loss.item():.6f}")
        return delta.detach()

    def frequency_guidance(self, grads, delta, H, W):
        fft_grads = [torch.fft.fft2(g) for g in grads]
        fft_stack = torch.stack(fft_grads, dim=0)
        mag_stack = torch.abs(fft_stack)
        grad_h = torch.diff(mag_stack, dim=3)[:, :, :, :, :-1]
        grad_w = torch.diff(mag_stack, dim=4)[:, :, :, :-1, :]
        smoothness = torch.mean(torch.sqrt(grad_h**2 + grad_w**2), dim=(1, 2, 3, 4))
        consistency_weight = torch.exp(-smoothness)
        consistency_weight_expanded = consistency_weight.view(self.num_model, 1, 1, 1, 1)
        weighted_fft = torch.sum(fft_stack * consistency_weight_expanded, dim=0)
        weight_sum = torch.sum(consistency_weight) + self.epsilon
        modulated_fft = weighted_fft / weight_sum
        device = modulated_fft.device
        freqs_h = torch.fft.fftfreq(H, d=1.0).to(device)
        freqs_w = torch.fft.fftfreq(W, d=1.0).to(device)
        grid_h, grid_w = torch.meshgrid(freqs_h, freqs_w, indexing='ij')
        freq_dist = torch.sqrt(grid_h**2 + grid_w**2)
        fft_delta = torch.fft.fft2(delta)
        freq_reg = torch.mean(torch.abs(fft_delta) * freq_dist.unsqueeze(0).unsqueeze(0))
        reg_fft = modulated_fft + self.freq_reg_weight * (freq_reg * fft_delta / (torch.abs(fft_delta) + 1e-8))
        total_grad = torch.real(torch.fft.ifft2(reg_fft))
        total_grad = self.spatial_smoothing(total_grad)
        return total_grad

    def spatial_smoothing(self, grad, kernel_size=3):
        if kernel_size == 3:
            kernel = torch.tensor([[1.0, 1.0, 1.0],
                                   [1.0, 2.0, 1.0],
                                   [1.0, 1.0, 1.0]], device=grad.device)
        else:
            raise ValueError("Unsupported kernel size")
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        B, C, H, W = grad.shape
        grad_reshaped = grad.view(B * C, 1, H, W)
        grad_smoothed = F.conv2d(grad_reshaped, kernel, padding=kernel_size // 2)
        grad_smoothed = grad_smoothed.view(B, C, H, W)
        return grad_smoothed

class EnsembleModel(torch.nn.Module):
    def __init__(self, models, mode='mean'):
        super().__init__()
        if not models:
            raise ValueError("No models provided for EnsembleModel")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = nn.ModuleList([model.to(self.device) for model in models])
        self.softmax = torch.nn.Softmax(dim=1)
        self.num_models = len(models)
        self.mode = mode

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        outputs = torch.stack(outputs, dim=0)
        if self.mode == 'mean':
            return torch.mean(outputs, dim=0)
        elif self.mode == 'ind':
            return outputs
        else:
            raise NotImplementedError

class AdvDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir=None, output_dir=None, targeted=False, eval=False):
        self.targeted = targeted
        self.data_dir = input_dir
        self.f2l = self.load_labels(os.path.join(self.data_dir, 'labels.csv'))
        if eval:
            self.data_dir = output_dir
            print(f'=> Eval mode: evaluating on {self.data_dir}')
        else:
            self.data_dir = os.path.join(self.data_dir, 'images')
            print(f'=> Train mode: training on {self.data_dir}')
            print(f'Save images to {output_dir}')

    def __len__(self):
        return len(self.f2l.keys())

    def __getitem__(self, idx):
        filename = list(self.f2l.keys())[idx]
        filepath = os.path.join(self.data_dir, filename)
        image = Image.open(filepath).resize((img_height, img_width)).convert('RGB')
        image = np.array(image).astype(np.float32)/255
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = self.f2l[filename]
        return image, label, filename

    def load_labels(self, file_name):
        dev = pd.read_csv(file_name)
        if self.targeted:
            f2l = {dev.iloc[i]['filename']: [dev.iloc[i]['label'], dev.iloc[i]['targeted_label']] for i in range(len(dev))}
        else:
            f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
        return f2l

def wrap_model(model):
    if hasattr(model, 'default_cfg'):
        mean = model.default_cfg['mean']
        std = model.default_cfg['std']
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)
    return torch.nn.Sequential(normalize, model)

def clamp(x, x_min, x_max):
    return torch.min(torch.max(x, x_min), x_max)

def save_images(output_dir, adversaries, filenames):
    adversaries = (adversaries.detach().permute((0,2,3,1)).cpu().numpy() * 255).astype(np.uint8)
    for i, filename in enumerate(filenames):
        Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename))
def get_parser():
    parser = argparse.ArgumentParser(description='Generating transferable adversarial examples')
    parser.add_argument('-e', '--eval', action='store_true', help='attack/evaluation')
    parser.add_argument('--attack', default='FSEA', type=str, help='the attack algorithm')
    parser.add_argument('--epoch', default=10, type=int, help='the iterations for updating the adversarial patch')
    parser.add_argument('--batchsize', default=8, type=int, help='the batch size')
    parser.add_argument('--eps', default=16/255, type=float, help='the perturbation budget')
    parser.add_argument('--alpha', default=1.6/255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--momentum', default=0., type=float, help='the decay factor for momentum based attack')
    parser.add_argument('--model', default='resnet18', type=str, help='the source surrogate model(s), comma-separated for ensemble')
    parser.add_argument('--ensemble', action='store_true', help='enable ensemble attack')
    parser.add_argument('--random_start', default=True, type=bool, help='set random start')
    parser.add_argument('--input_dir', default='./data', type=str, help='the path for custom benign images')
    parser.add_argument('--output_dir', default='./results', type=str, help='the path to store the adversarial patches')
    parser.add_argument('--targeted', action='store_true', help='targeted attack')
    parser.add_argument('--GPU_ID', default='1', type=str, help='CUDA device ID')
    return parser

def main():
    args = get_parser().parse_args()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dataset = AdvDataset(input_dir=args.input_dir, output_dir=args.output_dir, targeted=args.targeted, eval=args.eval)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=4)

    model_names = args.model.split(',') if args.ensemble or ',' in args.model else args.model
    attacker = FSEA(model_name=model_names, epsilon=args.eps, alpha=args.alpha, epoch=args.epoch, 
                    decay=args.momentum, targeted=args.targeted, random_start=args.random_start)

    if not args.eval:
        for batch_idx, (images, labels, filenames) in tqdm.tqdm(enumerate(dataloader)):
            perturbations = attacker(images, labels)
            save_images(args.output_dir, images + perturbations.cpu(), filenames)
            print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            print(f"Cached memory: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
            torch.cuda.empty_cache()
    else:
        asr = {}
        res = '|'
        for model_name, model in load_pretrained_model(cnn_model_paper, vit_model_paper):
            model = wrap_model(model.eval().cuda())
            for p in model.parameters():
                p.requires_grad = False
            correct, total = 0, 0
            for images, labels, _ in dataloader:
                if args.targeted:
                    labels = labels[1]
                pred = model(images.cuda())
                correct += (labels.numpy() == pred.argmax(dim=1).detach().cpu().numpy()).sum()
                total += labels.shape[0]
            asr[model_name] = (correct / total * 100) if args.targeted else ((1 - correct / total) * 100)
            print(f"{model_name}: {asr[model_name]:.1f}")
            res += f' {asr[model_name]:.1f} |'
        print(asr)
        print(res)
        with open('results_eval.txt', 'a') as f:
            f.write(args.output_dir + res + '\n')

cnn_model_paper = ['inception_v3', 'inception_v4', 'resnetv2_50x1_bitm', 'resnet18', 'resnet101', 
                   'resnext50_32x4d', 'densenet121', 'resnet50', 'wide_resnet50_2', 'resnetv2_101x1_bitm']
vit_model_paper = ['vit_tiny_patch16_224', 'deit_tiny_patch16_224', 'swin_tiny_patch4_window7_224', 
                   'vit_small_patch16_224', 'vit_base_patch16_224', 'pit_b_224', 'visformer_small', 
                   'deit_base_patch16_224', 'swin_base_patch4_window7_224']

def load_pretrained_model(cnn_model=[], vit_model=[]):
    for model_name in cnn_model:
        yield model_name, timm.create_model(model_name, pretrained=True)
    for model_name in vit_model:
        yield model_name, timm.create_model(model_name, pretrained=True)

if __name__ == '__main__':
    main()