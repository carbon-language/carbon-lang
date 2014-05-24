; RUN: llc < %s -march=arm64 -aarch64-neon-syntax=apple | FileCheck %s

; CHECK: fptosi_1
; CHECK: fcvtzs.2d
; CHECK: xtn.2s
; CHECK: ret
define void @fptosi_1() nounwind noinline ssp {
entry:
  %0 = fptosi <2 x double> undef to <2 x i32>
  store <2 x i32> %0, <2 x i32>* undef, align 8
  ret void
}

; CHECK: fptoui_1
; CHECK: fcvtzu.2d
; CHECK: xtn.2s
; CHECK: ret
define void @fptoui_1() nounwind noinline ssp {
entry:
  %0 = fptoui <2 x double> undef to <2 x i32>
  store <2 x i32> %0, <2 x i32>* undef, align 8
  ret void
}

