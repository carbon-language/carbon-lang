;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; CHECK: @main
; CHECK: ADD *

define void @main() #0 {
main_body:
  %0 = call float @llvm.R600.load.input(i32 4)
  %1 = call float @llvm.R600.load.input(i32 5)
  %2 = call float @llvm.R600.load.input(i32 6)
  %3 = call float @llvm.R600.load.input(i32 7)
  %4 = call float @llvm.R600.load.input(i32 8)
  %5 = fadd float %0, 2.0
  %6 = fadd float %1, 3.0
  %7 = fadd float %2, 4.0
  %8 = fadd float %3, 5.0
  %9 = bitcast float %4 to i32
  %10 = mul i32 %9, 6
  %11 = bitcast i32 %10 to float
  %12 = insertelement <4 x float> undef, float %5, i32 0
  %13 = insertelement <4 x float> %12, float %6, i32 1
  %14 = insertelement <4 x float> %13, float %7, i32 2
  %15 = insertelement <4 x float> %14, float %8, i32 3
  %16 = insertelement <4 x float> %15, float %11, i32 3

  %17 = call float @llvm.AMDGPU.dp4(<4 x float> %15,<4 x float> %16)
  %18 = insertelement <4 x float> undef, float %17, i32 0
  call void @llvm.R600.store.swizzle(<4 x float> %18, i32 0, i32 2)
  ret void
}

; CHECK: @main
; CHECK-NOT: ADD *

define void @main2() #0 {
main_body:
  %0 = call float @llvm.R600.load.input(i32 4)
  %1 = call float @llvm.R600.load.input(i32 5)
  %2 = call float @llvm.R600.load.input(i32 6)
  %3 = call float @llvm.R600.load.input(i32 7)
  %4 = call float @llvm.R600.load.input(i32 8)
  %5 = fadd float %0, 2.0
  %6 = fadd float %1, 3.0
  %7 = fadd float %2, 4.0
  %8 = fadd float %3, 2.0
  %9 = bitcast float %4 to i32
  %10 = mul i32 %9, 6
  %11 = bitcast i32 %10 to float
  %12 = insertelement <4 x float> undef, float %5, i32 0
  %13 = insertelement <4 x float> %12, float %6, i32 1
  %14 = insertelement <4 x float> %13, float %7, i32 2
  %15 = insertelement <4 x float> %14, float %8, i32 3
  %16 = insertelement <4 x float> %15, float %11, i32 3

  %17 = call float @llvm.AMDGPU.dp4(<4 x float> %15,<4 x float> %16)
  %18 = insertelement <4 x float> undef, float %17, i32 0
  call void @llvm.R600.store.swizzle(<4 x float> %18, i32 0, i32 2)
  ret void
}

; Function Attrs: readnone
declare float @llvm.R600.load.input(i32) #1
declare float @llvm.AMDGPU.dp4(<4 x float>, <4 x float>) #1

declare void @llvm.R600.store.swizzle(<4 x float>, i32, i32)

attributes #0 = { "ShaderType"="1" }
attributes #1 = { readnone }
