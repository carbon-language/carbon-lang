; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-unknown-unknown"

; Verify that both %input and %output are converted to global pointers and then
; addrspacecast'ed back to the original type.
define void @kernel(float* %input, float* %output) {
; CHECK-LABEL: .visible .entry kernel(
; CHECK: cvta.to.global.u64
; CHECK: cvta.to.global.u64
  %1 = load float, float* %input, align 4
; CHECK: ld.global.f32
  store float %1, float* %output, align 4
; CHECK: st.global.f32
  ret void
}

define void @kernel2(float addrspace(1)* %input, float addrspace(1)* %output) {
; CHECK-LABEL: .visible .entry kernel2(
; CHECK-NOT: cvta.to.global.u64
  %1 = load float, float addrspace(1)* %input, align 4
; CHECK: ld.global.f32
  store float %1, float addrspace(1)* %output, align 4
; CHECK: st.global.f32
  ret void
}

!nvvm.annotations = !{!0, !1}
!0 = !{void (float*, float*)* @kernel, !"kernel", i32 1}
!1 = !{void (float addrspace(1)*, float addrspace(1)*)* @kernel2, !"kernel", i32 1}
