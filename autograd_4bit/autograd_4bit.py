"""GPTQv2 AutogradMatmul4bit

References: 
1. Original AutogradMatmul4bit implementation - https://github.com/johnsmith0031/alpaca_lora_4bit/blob/main/autograd_4bit.py
2. Forward and backward kernels - https://github.com/qwopqwop200/GPTQ-for-LLaMa/blob/triton/quant.py
"""
import torch
import torch.nn as nn
import math
import triton
from autograd_4bit.triton_kernels import matmul_248_kernel, trans_matmul_248_kernel


class AutogradMatmul4bit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, qweight, scales, qzeros, g_idx, bits, maxq):
        output = torch.empty((input.shape[0], qweight.shape[1]), device='cuda', dtype=torch.float16)
        grid = lambda META: (triton.cdiv(input.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(qweight.shape[1], META['BLOCK_SIZE_N']),)
        matmul_248_kernel[grid](input, qweight, output,
                                scales, qzeros, g_idx,
                                input.shape[0], qweight.shape[1], input.shape[1], bits, maxq,
                                input.stride(0), input.stride(1),
                                qweight.stride(0), qweight.stride(1),
                                output.stride(0), output.stride(1),
                                scales.stride(0), qzeros.stride(0))

        ctx.save_for_backward(qweight, scales, qzeros, g_idx)
        ctx.input_shape, ctx.bits,ctx.maxq = input.shape,bits, maxq
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_shape, bits, maxq = ctx.input_shape, ctx.bits, ctx.maxq
        qweight, scales, qzeros, g_idx = ctx.saved_tensors
        grade_input = None

        if ctx.needs_input_grad[0]:
            grade_input = torch.empty((input_shape[0], input_shape[1]), device='cuda', dtype=torch.float32)
            grid = lambda META: (triton.cdiv(input_shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(input_shape[1], META['BLOCK_SIZE_K']),)
            trans_matmul_248_kernel[grid](grad_output, qweight, grade_input,
                                          scales, qzeros, g_idx,
                                          input_shape[0], qweight.shape[1], input_shape[1], bits, maxq,
                                          grad_output.stride(0), grad_output.stride(1),
                                          qweight.stride(0), qweight.stride(1),
                                          grade_input.stride(0), grade_input.stride(1),
                                          scales.stride(0), qzeros.stride(0))
        return grade_input, None, None, None, None, None, None


class Autograd4bitQuantLinear(nn.Module):

    def __init__(self, in_features, out_features, groupsize, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = 4  # Hardcoded 4-bits quantizations
        self.maxq = 2 ** self.bits - 1
        self.groupsize = groupsize if groupsize != -1 else in_features
        
        self.register_buffer('qweight', torch.zeros((in_features // 32 * self.bits, out_features), dtype=torch.int32))
        self.register_buffer('qzeros', torch.zeros((math.ceil(in_features / self.groupsize), out_features // 32 * self.bits), dtype=torch.int32))
        self.register_buffer('scales', torch.zeros((math.ceil(in_features / self.groupsize), out_features), dtype=torch.float16))
        self.register_buffer('g_idx', torch.tensor([i // self.groupsize  for i in range(in_features)], dtype = torch.int32))
        if bias:
            self.register_buffer('bias', torch.zeros(out_features,dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        out = AutogradMatmul4bit.apply(x.reshape(-1,x.shape[-1]), self.qweight, self.scales, 
                                        self.qzeros, self.g_idx, self.bits, self.maxq)
        out = out + self.bias if self.bias is not None else out  
        return out.reshape(out_shape)


def make_quant_for_4bit_autograd(module, names, name='', groupsize=-1):
    if isinstance(module, Autograd4bitQuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(
                module, attr, Autograd4bitQuantLinear(tmp.in_features, tmp.out_features, groupsize=groupsize)
            )
    for name1, child in module.named_children():
        make_quant_for_4bit_autograd(child, names, name + '.' + name1 if name != '' else name1, groupsize=groupsize)
