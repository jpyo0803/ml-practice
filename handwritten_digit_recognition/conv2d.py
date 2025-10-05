import torch
import torch.nn as nn

class MyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Kernel을 Im2col 기법에 맞게 변경
        self.kernel = nn.Parameter(
            torch.randn(out_channels, in_channels * kernel_size * kernel_size) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))


    def forward(self, x):
        # Convolution 연산 수행
        # x = x.to(torch.device('cuda'))
        batch_size, in_channels, height, width = x.shape

        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        if self.padding > 0:
            x = nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding))

        height, width = x.shape[2], x.shape[3]

        # im2col 기법을 사용하여 연산 최적화

        '''
            Convolution 원본
        '''
        # for b in range(batch_size):
        #     for oc in range(self.out_channels):
        #         for oh in range(out_height):
        #             for ow in range(out_width):
        #                 h_start = oh * self.stride
        #                 w_start = ow * self.stride
        #                 h_end = h_start + self.kernel_size
        #                 w_end = w_start + self.kernel_size

        #                 res = torch.dot(x[b, :, h_start:h_end, w_start:w_end].flatten(),
        #                                 self.kernel[oc].flatten())
        #                 out[b, oc, oh, ow] = res + self.bias[oc]

        '''
            Naive Im2col 구현
        '''
        # x_cols = torch.zeros((batch_size, in_channels * self.kernel_size * self.kernel_size, out_height * out_width), device=x.device)

        # for oh in range(out_height):
        #     for ow in range(out_width):
        #         h_start = oh * self.stride
        #         w_start = ow * self.stride
        #         h_end = h_start + self.kernel_size
        #         w_end = w_start + self.kernel_size

        #         x_cols[:, :, oh * out_width + ow] = x[:, :, h_start:h_end, w_start:w_end].reshape(batch_size, -1)

        '''
            개선된 Im2col 구현 (위의 Naive Im2col과 동일한 결과)
        '''
        x_cols = torch.nn.functional.unfold(
            x, kernel_size=self.kernel_size, stride=self.stride
        )  # (batch_size, in_channels * kernel_size * kernel_size, out_height * out_width)


        out = torch.matmul(self.kernel, x_cols) + self.bias.unsqueeze(1)
        out = out.view(batch_size, self.out_channels, out_height, out_width)

        return out