; RUN: llc < %s -march=r600 -show-mc-encoding  -mcpu=rv710 | FileCheck %s

; CHECK: TEX 9 @4 ;  encoding: [0x04,0x00,0x00,0x00,0x00,0x04,0x88,0x80]

define void @test(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in) {
   %1 = call float @llvm.R600.load.input(i32 4)
   %2 = call float @llvm.R600.load.input(i32 5)
   %3 = call float @llvm.R600.load.input(i32 6)
   %4 = call float @llvm.R600.load.input(i32 7)
   %5 = insertelement <4 x float> undef, float %1, i32 0
   %6 = insertelement <4 x float> %5, float %2, i32 1
   %7 = insertelement <4 x float> %6, float %3, i32 2
   %8 = insertelement <4 x float> %7, float %4, i32 3
   %9 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %8, i32 0, i32 0, i32 1)
   %10 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %8, i32 1, i32 0, i32 1)
   %11 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %8, i32 2, i32 0, i32 1)
   %12 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %8, i32 3, i32 0, i32 1)
   %13 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %8, i32 4, i32 0, i32 1)
   %14 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %8, i32 5, i32 0, i32 1)
   %15 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %8, i32 6, i32 0, i32 1)
   %16 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %8, i32 7, i32 0, i32 1)
   %17 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %8, i32 8, i32 0, i32 1)
   %18 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %8, i32 9, i32 0, i32 1)
   %19 = fadd <4 x float> %9, %10
   %20 = fadd <4 x float> %19, %11
   %21 = fadd <4 x float> %20, %12
   %22 = fadd <4 x float> %21, %13
   %23 = fadd <4 x float> %22, %14
   %24 = fadd <4 x float> %23, %15
   %25 = fadd <4 x float> %24, %16
   %26 = fadd <4 x float> %25, %17
   %27 = fadd <4 x float> %26, %18
   call void @llvm.R600.store.swizzle(<4 x float> %27, i32 0, i32 2)
   ret void
}

declare <4 x float> @llvm.AMDGPU.tex(<4 x float>, i32, i32, i32) readnone

; Function Attrs: readnone
declare float @llvm.R600.load.input(i32) #1


declare void @llvm.R600.store.swizzle(<4 x float>, i32, i32)
attributes #1 = { readnone }
