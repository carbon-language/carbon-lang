; RUN: opt < %s -S -inline -inline-threshold=2 | FileCheck %s
; RUN: opt < %s -S -strip-debug -inline -inline-threshold=2 | FileCheck %s
;
; The purpose of this test is to check that debug info doesn't influence
; inlining decisions.


target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.dbg.declare(metadata, metadata) #1
declare void @llvm.dbg.value(metadata, i64, metadata) #1

define <4 x float> @inner_vectors(<4 x float> %a, <4 x float> %b) {
entry:
  call void @llvm.dbg.value(metadata !{}, i64 0, metadata !{})
  %mul = fmul <4 x float> %a, <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>
  call void @llvm.dbg.value(metadata !{}, i64 0, metadata !{})
  %mul1 = fmul <4 x float> %b, <float 5.000000e+00, float 5.000000e+00, float 5.000000e+00, float 5.000000e+00>
  call void @llvm.dbg.value(metadata !{}, i64 0, metadata !{})
  %add = fadd <4 x float> %mul, %mul1
  ret <4 x float> %add
}

define float @outer_vectors(<4 x float> %a, <4 x float> %b) {
; CHECK-LABEL: @outer_vectors(
; CHECK-NOT: call <4 x float> @inner_vectors(
; CHECK: ret float

entry:
  call void @llvm.dbg.value(metadata !{}, i64 0, metadata !{})
  call void @llvm.dbg.value(metadata !{}, i64 0, metadata !{})
  %call = call <4 x float> @inner_vectors(<4 x float> %a, <4 x float> %b)
  call void @llvm.dbg.value(metadata !{}, i64 0, metadata !{})
  %vecext = extractelement <4 x float> %call, i32 0
  %vecext1 = extractelement <4 x float> %call, i32 1
  %add = fadd float %vecext, %vecext1
  %vecext2 = extractelement <4 x float> %call, i32 2
  %add3 = fadd float %add, %vecext2
  %vecext4 = extractelement <4 x float> %call, i32 3
  %add5 = fadd float %add3, %vecext4
  ret float %add5
}

attributes #0 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !{}, metadata !2, metadata !2, metadata !""}
!1 = metadata !{metadata !"", metadata !""}
!2 = metadata !{i32 0}
!3 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!4 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!5 = metadata !{metadata !""}
