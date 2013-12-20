; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; CHECK-LABEL: @trunc_f32
; CHECK: TRUNC

define void @trunc_f32(float addrspace(1)* %out, float %in) {
entry:
  %0 = call float @llvm.trunc.f32(float %in)
  store float %0, float  addrspace(1)* %out
  ret void
}

declare float @llvm.trunc.f32(float)
