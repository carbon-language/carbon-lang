; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

target triple = "nvptx-unknown-nvcl"

; CHECK-LABEL: .entry foo(
define void @foo(i64 %img, i64 %sampler, <5 x float>* %v) {
; The parameter alignment should be the next power of 2 of 5xsizeof(float),
; which is 32.
; CHECK: .param .u32 .ptr .align 32 foo_param_2
  ret void
}

!nvvm.annotations = !{!1, !2, !3}
!1 = !{void (i64, i64, <5 x float>*)* @foo, !"kernel", i32 1}
!2 = !{void (i64, i64, <5 x float>*)* @foo, !"rdoimage", i32 0}
!3 = !{void (i64, i64, <5 x float>*)* @foo, !"sampler", i32 1}
