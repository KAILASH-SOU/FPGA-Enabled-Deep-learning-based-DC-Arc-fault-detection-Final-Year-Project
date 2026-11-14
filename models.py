import torch
import torch.nn as nn
import torch.nn.functional as F

class FakeQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, qmin, qmax):
        ctx.save_for_backward(x, scale)
        ctx.qmin = qmin
        ctx.qmax = qmax
        q_x = torch.round(x / scale + zero_point)
        q_x = torch.clamp(q_x, qmin, qmax)
        dq_x = (q_x - zero_point) * scale
        return dq_x

    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        qmin = ctx.qmin
        qmax = ctx.qmax
        normalized = x / scale
        mask = (normalized >= qmin) & (normalized <= qmax)
        grad_input = grad_output * mask.float()
        return grad_input, None, None, None, None

class FakeQuantizer(nn.Module):
    def __init__(self, bit_width=8, symmetric=True):
        super(FakeQuantizer, self).__init__()
        self.bit_width = bit_width
        self.symmetric = symmetric
        if symmetric:
            self.qmin = -(2 ** (bit_width - 1))
            self.qmax = (2 ** (bit_width - 1)) - 1
        else:
            self.qmin = 0
            self.qmax = (2 ** bit_width) - 1

        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0))
        self.enabled = True

    def update_scale(self, x):
        with torch.no_grad():
            if self.symmetric:
                max_val = torch.max(torch.abs(x))
                max_val = torch.clamp(max_val, min=1e-6)
                self.scale = max_val / float(self.qmax)
                self.zero_point = torch.tensor(0.0, device=x.device)
            else:
                min_val = torch.min(x)
                max_val = torch.max(x)
                diff = torch.clamp(max_val - min_val, min=1e-6)
                self.scale = diff / float(self.qmax - self.qmin)
                self.zero_point = torch.round(-min_val / self.scale)

    def forward(self, x):
        if not self.enabled:
            return x
        if self.training:
            self.update_scale(x)
        return FakeQuantizeSTE.apply(x, self.scale, self.zero_point, self.qmin, self.qmax)

class ArcFaultCNN_FPGA(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, dropout_rate=0.2):
        super(ArcFaultCNN_FPGA, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.act1 = nn.ReLU6()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU6()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.act3 = nn.ReLU6()
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.fc1 = nn.Linear(32 * 20, 32)
        self.act4 = nn.ReLU6()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(32, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x = self.pool2(self.act2(self.bn2(self.conv2(x))))
        x = self.pool3(self.act3(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.act4(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

class ArcFaultCNN_QAT(nn.Module):
    def __init__(self, fp32_model=None):
        super(ArcFaultCNN_QAT, self).__init__()
        self.quant_input = FakeQuantizer(bit_width=8, symmetric=False)

        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.quant_w1 = FakeQuantizer(bit_width=8, symmetric=True)
        self.bn1 = nn.BatchNorm1d(16)
        self.act1 = nn.ReLU6()
        self.quant_a1 = FakeQuantizer(bit_width=8, symmetric=False)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.quant_w2 = FakeQuantizer(bit_width=8, symmetric=True)
        self.bn2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU6()
        self.quant_a2 = FakeQuantizer(bit_width=8, symmetric=False)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.quant_w3 = FakeQuantizer(bit_width=8, symmetric=True)
        self.bn3 = nn.BatchNorm1d(32)
        self.act3 = nn.ReLU6()
        self.quant_a3 = FakeQuantizer(bit_width=8, symmetric=False)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.fc1 = nn.Linear(32 * 20, 32)
        self.quant_w_fc1 = FakeQuantizer(bit_width=8, symmetric=True)
        self.act4 = nn.ReLU6()
        self.quant_a4 = FakeQuantizer(bit_width=8, symmetric=False)

        self.fc2 = nn.Linear(32, 1)
        self.quant_w_fc2 = FakeQuantizer(bit_width=8, symmetric=True)
        self.sigmoid = nn.Sigmoid()

        if fp32_model is not None:
            self.load_from_fp32(fp32_model)

    def load_from_fp32(self, fp32_model):
        self.conv1.load_state_dict(fp32_model.conv1.state_dict())
        self.bn1.load_state_dict(fp32_model.bn1.state_dict())
        self.conv2.load_state_dict(fp32_model.conv2.state_dict())
        self.bn2.load_state_dict(fp32_model.bn2.state_dict())
        self.conv3.load_state_dict(fp32_model.conv3.state_dict())
        self.bn3.load_state_dict(fp32_model.bn3.state_dict())
        self.fc1.load_state_dict(fp32_model.fc1.state_dict())
        self.fc2.load_state_dict(fp32_model.fc2.state_dict())

    def forward(self, x):
        x = self.quant_input(x)

        w1_q = self.quant_w1(self.conv1.weight)
        x = F.conv1d(x, w1_q, self.conv1.bias, stride=self.conv1.stride, padding=self.conv1.padding)
        x = self.pool1(self.quant_a1(self.act1(self.bn1(x))))

        w2_q = self.quant_w2(self.conv2.weight)
        x = F.conv1d(x, w2_q, self.conv2.bias, stride=self.conv2.stride, padding=self.conv2.padding)
        x = self.pool2(self.quant_a2(self.act2(self.bn2(x))))

        w3_q = self.quant_w3(self.conv3.weight)
        x = F.conv1d(x, w3_q, self.conv3.bias, stride=self.conv3.stride, padding=self.conv3.padding)
        x = self.pool3(self.quant_a3(self.act3(self.bn3(x))))

        x = x.view(x.size(0), -1)
        w_fc1_q = self.quant_w_fc1(self.fc1.weight)
        x = F.linear(x, w_fc1_q, self.fc1.bias)
        x = self.quant_a4(self.act4(x))

        w_fc2_q = self.quant_w_fc2(self.fc2.weight)
        x = F.linear(x, w_fc2_q, self.fc2.bias)
        x = self.sigmoid(x)
        return x

class ArcFaultCNN_Legacy(nn.Module):
    def __init__(self):
        super(ArcFaultCNN_Legacy, self).__init__()
        self.conv1 = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(160, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
