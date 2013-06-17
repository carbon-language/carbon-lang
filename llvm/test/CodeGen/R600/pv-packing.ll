; RUN: llc < %s -march=r600 -mcpu=cayman | FileCheck %s

;CHECK: DOT4  T{{[0-9]\.X}}
;CHECK: MULADD_IEEE * T{{[0-9]\.W}}

define void @main() #0 {
main_body:
  %0 = call float @llvm.R600.load.input(i32 4)
  %1 = call float @llvm.R600.load.input(i32 5)
  %2 = call float @llvm.R600.load.input(i32 6)
  %3 = call float @llvm.R600.load.input(i32 8)
  %4 = call float @llvm.R600.load.input(i32 9)
  %5 = call float @llvm.R600.load.input(i32 10)
  %6 = call float @llvm.R600.load.input(i32 12)
  %7 = call float @llvm.R600.load.input(i32 13)
  %8 = call float @llvm.R600.load.input(i32 14)
  %9 = load <4 x float> addrspace(8)* null
  %10 = load <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>] addrspace(8)* null, i64 0, i32 1)
  %11 = call float @llvm.AMDGPU.dp4(<4 x float> %9, <4 x float> %9)
  %12 = fmul float %0, %3
  %13 = fadd float %12, %6
  %14 = fmul float %1, %4
  %15 = fadd float %14, %7
  %16 = fmul float %2, %5
  %17 = fadd float %16, %8
  %18 = fmul float %11, %11
  %19 = fadd float %18, %0
  %20 = insertelement <4 x float> undef, float %13, i32 0
  %21 = insertelement <4 x float> %20, float %15, i32 1
  %22 = insertelement <4 x float> %21, float %17, i32 2
  %23 = insertelement <4 x float> %22, float %19, i32 3
  %24 = call float @llvm.AMDGPU.dp4(<4 x float> %23, <4 x float> %10)
  %25 = insertelement <4 x float> undef, float %24, i32 0
  call void @llvm.R600.store.swizzle(<4 x float> %25, i32 0, i32 2)
  ret void
}

; Function Attrs: readnone
declare float @llvm.R600.load.input(i32) #1

; Function Attrs: readnone
declare float @llvm.AMDGPU.dp4(<4 x float>, <4 x float>) #1


declare void @llvm.R600.store.swizzle(<4 x float>, i32, i32)

attributes #0 = { "ShaderType"="1" }
attributes #1 = { readnone }
attributes #2 = { readonly }
attributes #3 = { nounwind readonly }
