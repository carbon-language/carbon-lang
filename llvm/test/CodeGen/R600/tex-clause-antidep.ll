;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

;CHECK: TEX
;CHECK-NEXT: ALU

define void @test() {
  %1 = call float @llvm.R600.load.input(i32 0)
  %2 = call float @llvm.R600.load.input(i32 1)
  %3 = call float @llvm.R600.load.input(i32 2)
  %4 = call float @llvm.R600.load.input(i32 3)
  %5 = insertelement <4 x float> undef, float %1, i32 0
  %6 = insertelement <4 x float> %5, float %2, i32 1
  %7 = insertelement <4 x float> %6, float %3, i32 2
  %8 = insertelement <4 x float> %7, float %4, i32 3
  %9 = call <4 x float> @llvm.R600.tex(<4 x float> %8, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %10 = call <4 x float> @llvm.R600.tex(<4 x float> %8, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %11 = fadd <4 x float> %9, %10
  call void @llvm.R600.store.swizzle(<4 x float> %11, i32 0, i32 0)
  ret void
}

declare float @llvm.R600.load.input(i32) readnone
declare <4 x float> @llvm.R600.tex(<4 x float>, i32, i32, i32, i32, i32, i32, i32, i32, i32) readnone
declare void @llvm.R600.store.swizzle(<4 x float>, i32, i32)
