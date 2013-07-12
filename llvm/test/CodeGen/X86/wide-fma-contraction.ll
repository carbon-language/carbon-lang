; RUN: llc -march=x86 -mcpu=bdver2 -mattr=-fma -mtriple=x86_64-apple-darwin < %s | FileCheck %s
; RUN: llc -march=x86 -mcpu=bdver2 -mattr=-fma,-fma4 -mtriple=x86_64-apple-darwin < %s | FileCheck %s --check-prefix=CHECK-NOFMA

; CHECK-LABEL: fmafunc
; CHECK-NOFMA-LABEL: fmafunc
define <16 x float> @fmafunc(<16 x float> %a, <16 x float> %b, <16 x float> %c) {

; CHECK-NOT: vmulps
; CHECK-NOT: vaddps
; CHECK: vfmaddps
; CHECK-NOT: vmulps
; CHECK-NOT: vaddps
; CHECK: vfmaddps
; CHECK-NOT: vmulps
; CHECK-NOT: vaddps

; CHECK-NOFMA-NOT: calll
; CHECK-NOFMA: vmulps
; CHECK-NOFMA: vaddps
; CHECK-NOFMA-NOT: calll
; CHECK-NOFMA: vmulps
; CHECK-NOFMA: vaddps
; CHECK-NOFMA-NOT: calll

  %ret = tail call <16 x float> @llvm.fmuladd.v16f32(<16 x float> %a, <16 x float> %b, <16 x float> %c)
  ret <16 x float> %ret
}

declare <16 x float> @llvm.fmuladd.v16f32(<16 x float>, <16 x float>, <16 x float>) nounwind readnone
