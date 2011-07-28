; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; CHECK: vcvtdq2ps %ymm
define <8 x float> @funcA(<8 x i32> %a) nounwind {
  %b = sitofp <8 x i32> %a to <8 x float>
  ret <8 x float> %b
}

; CHECK: vcvttps2dq %ymm
define <8 x i32> @funcB(<8 x float> %a) nounwind {
  %b = fptosi <8 x float> %a to <8 x i32>
  ret <8 x i32> %b
}

