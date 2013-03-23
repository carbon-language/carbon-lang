; RUN: llc -march=x86 -mattr=+fma4 -mtriple=x86_64-apple-darwin < %s | FileCheck %s

; CHECK: fmafunc
define <16 x float> @fmafunc(<16 x float> %a, <16 x float> %b, <16 x float> %c) {
; CHECK-NOT: vmulps
; CHECK-NOT: vaddps
; CHECK: vfmaddps
; CHECK-NOT: vmulps
; CHECK-NOT: vaddps
; CHECK: vfmaddps
; CHECK-NOT: vmulps
; CHECK-NOT: vaddps
  %ret = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %a, <16 x float> %b, <16 x float> %c)
  ret <16 x float> %ret
}

declare <16 x float> @llvm.fmuladd.v16f32(<16 x float>, <16 x float>, <16 x float>) nounwind readnone



