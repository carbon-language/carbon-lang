; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

target triple = "nvptx-nvidia-cuda"

; CHECK: .visible .global .align 16 .b8 sphPosArr[80] = {0, 0, 192, 192, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 64, 192, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 64, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 192, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63};
@sphPosArr = constant [5 x <4 x float>] [<4 x float> <float -6.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, <4 x float> <float -3.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, <4 x float> <float 3.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, <4 x float> <float 6.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>], align 16

