; RUN: llc < %s -mcpu=cortex-a8 | FileCheck %s
target triple = "thumbv7-apple-ios"

; CHECK: local_split
;
; The load must go into d0-15 which are all clobbered by the asm.
; RAGreedy should split the range and use d16-d31 to avoid a spill.
;
; CHECK: vldr s
; CHECK-NOT: vstr
; CHECK: vadd.f32
; CHECK-NOT: vstr
; CHECK: vorr
; CHECK: vstr s
define void @local_split(float* nocapture %p) nounwind ssp {
entry:
  %x = load float* %p, align 4
  %a = fadd float %x, 1.0
  tail call void asm sideeffect "", "~{d0},~{d1},~{d2},~{d3},~{d4},~{d5},~{d6},~{d7},~{d8},~{d9},~{d10},~{d11},~{d12},~{d13},~{d14},~{d15}"() nounwind
  store float %a, float* %p, align 4
  ret void
}
