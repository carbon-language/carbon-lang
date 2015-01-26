; RUN: llc < %s -mtriple=x86_64-apple-macosx -mattr=+sse2 -mcpu=nehalem | FileCheck %s

; rdar: 12558838
; PR14221
; There is a mismatch between the intrinsic and the actual instruction.
; The actual instruction has a partial update of dest, while the intrinsic
; passes through the upper FP values. Here, we make sure the source and
; destination of each scalar unary op are the same.

define void @rsqrtss(<4 x float> %a) nounwind uwtable ssp {
entry:
; CHECK-LABEL: rsqrtss:
; CHECK: rsqrtss %xmm0, %xmm0
; CHECK-NEXT: cvtss2sd %xmm0
; CHECK-NEXT: movshdup
; CHECK-NEXT: cvtss2sd %xmm0
; CHECK-NEXT: movap
; CHECK-NEXT: jmp

  %0 = tail call <4 x float> @llvm.x86.sse.rsqrt.ss(<4 x float> %a) nounwind
  %a.addr.0.extract = extractelement <4 x float> %0, i32 0
  %conv = fpext float %a.addr.0.extract to double
  %a.addr.4.extract = extractelement <4 x float> %0, i32 1
  %conv3 = fpext float %a.addr.4.extract to double
  tail call void @callee(double %conv, double %conv3) nounwind
  ret void
}
declare void @callee(double, double)
declare <4 x float> @llvm.x86.sse.rsqrt.ss(<4 x float>) nounwind readnone

define void @rcpss(<4 x float> %a) nounwind uwtable ssp {
entry:
; CHECK-LABEL: rcpss:
; CHECK: rcpss %xmm0, %xmm0
; CHECK-NEXT: cvtss2sd %xmm0
; CHECK-NEXT: movshdup
; CHECK-NEXT: cvtss2sd %xmm0
; CHECK-NEXT: movap
; CHECK-NEXT: jmp

  %0 = tail call <4 x float> @llvm.x86.sse.rcp.ss(<4 x float> %a) nounwind
  %a.addr.0.extract = extractelement <4 x float> %0, i32 0
  %conv = fpext float %a.addr.0.extract to double
  %a.addr.4.extract = extractelement <4 x float> %0, i32 1
  %conv3 = fpext float %a.addr.4.extract to double
  tail call void @callee(double %conv, double %conv3) nounwind
  ret void
}
declare <4 x float> @llvm.x86.sse.rcp.ss(<4 x float>) nounwind readnone

define void @sqrtss(<4 x float> %a) nounwind uwtable ssp {
entry:
; CHECK-LABEL: sqrtss:
; CHECK: sqrtss %xmm0, %xmm0
; CHECK-NEXT: cvtss2sd %xmm0
; CHECK-NEXT: movshdup
; CHECK-NEXT: cvtss2sd %xmm0
; CHECK-NEXT: movap
; CHECK-NEXT: jmp

  %0 = tail call <4 x float> @llvm.x86.sse.sqrt.ss(<4 x float> %a) nounwind
  %a.addr.0.extract = extractelement <4 x float> %0, i32 0
  %conv = fpext float %a.addr.0.extract to double
  %a.addr.4.extract = extractelement <4 x float> %0, i32 1
  %conv3 = fpext float %a.addr.4.extract to double
  tail call void @callee(double %conv, double %conv3) nounwind
  ret void
}
declare <4 x float> @llvm.x86.sse.sqrt.ss(<4 x float>) nounwind readnone

define void @sqrtsd(<2 x double> %a) nounwind uwtable ssp {
entry:
; CHECK-LABEL: sqrtsd:
; CHECK: sqrtsd %xmm0, %xmm0
; CHECK-NEXT: cvtsd2ss %xmm0
; CHECK-NEXT: shufpd
; CHECK-NEXT: cvtsd2ss %xmm0
; CHECK-NEXT: movap
; CHECK-NEXT: jmp

 %0 = tail call <2 x double> @llvm.x86.sse2.sqrt.sd(<2 x double> %a) nounwind
 %a0 = extractelement <2 x double> %0, i32 0
 %conv = fptrunc double %a0 to float
 %a1 = extractelement <2 x double> %0, i32 1
 %conv3 = fptrunc double %a1 to float
 tail call void @callee2(float %conv, float %conv3) nounwind
 ret void
}

declare void @callee2(float, float)
declare <2 x double> @llvm.x86.sse2.sqrt.sd(<2 x double>) nounwind readnone

