; RUN: opt -S -basicaa -slp-vectorizer -dce < %s | FileCheck %s
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios5.0.0"

; CHECK-LABEL: @foo
define void @foo(float* noalias %a, float* noalias %b, float* noalias %c) {
entry:
; Check that we don't lose !nontemporal hint when vectorizing loads.
; CHECK: %{{[0-9]*}} = load <4 x float>, <4 x float>* %{{[0-9]+}}, align 4, !nontemporal !0
  %b1 = load float, float* %b, align 4, !nontemporal !0
  %arrayidx.1 = getelementptr inbounds float, float* %b, i64 1
  %b2 = load float, float* %arrayidx.1, align 4, !nontemporal !0
  %arrayidx.2 = getelementptr inbounds float, float* %b, i64 2
  %b3 = load float, float* %arrayidx.2, align 4, !nontemporal !0
  %arrayidx.3 = getelementptr inbounds float, float* %b, i64 3
  %b4 = load float, float* %arrayidx.3, align 4, !nontemporal !0

; Check that we don't introduce !nontemporal hint when the original scalar loads didn't have it.
; CHECK: %{{[0-9]*}} = load <4 x float>, <4 x float>* %{{[0-9]+}}, align 4{{$}}
  %c1 = load float, float* %c, align 4
  %arrayidx2.1 = getelementptr inbounds float, float* %c, i64 1
  %c2 = load float, float* %arrayidx2.1, align 4
  %arrayidx2.2 = getelementptr inbounds float, float* %c, i64 2
  %c3 = load float, float* %arrayidx2.2, align 4
  %arrayidx2.3 = getelementptr inbounds float, float* %c, i64 3
  %c4 = load float, float* %arrayidx2.3, align 4

  %a1 = fadd float %b1, %c1
  %a2 = fadd float %b2, %c2
  %a3 = fadd float %b3, %c3
  %a4 = fadd float %b4, %c4

; Check that we don't lose !nontemporal hint when vectorizing stores.
; CHECK: store <4 x float> %{{[0-9]+}}, <4 x float>* %{{[0-9]+}}, align 4, !nontemporal !0
  store float %a1, float* %a, align 4, !nontemporal !0
  %arrayidx3.1 = getelementptr inbounds float, float* %a, i64 1
  store float %a2, float* %arrayidx3.1, align 4, !nontemporal !0
  %arrayidx3.2 = getelementptr inbounds float, float* %a, i64 2
  store float %a3, float* %arrayidx3.2, align 4, !nontemporal !0
  %arrayidx3.3 = getelementptr inbounds float, float* %a, i64 3
  store float %a4, float* %arrayidx3.3, align 4, !nontemporal !0

; CHECK: ret void
  ret void
}

!0 = !{i32 1}
