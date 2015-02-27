; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=+avx < %s | FileCheck %s

; Verify that we're folding the load into the math instruction.

; FIXME: The folding should also happen without the avx attribute; 
; ie, when generating SSE (non-VEX-prefixed) instructions.

define float @rcpss(float* %a) {
; CHECK-LABEL: rcpss:
; CHECK:       vrcpss (%rdi), %xmm0, %xmm0

    %ld = load float, float* %a
    %ins = insertelement <4 x float> undef, float %ld, i32 0
    %res = tail call <4 x float> @llvm.x86.sse.rcp.ss(<4 x float> %ins)
    %ext = extractelement <4 x float> %res, i32 0
    ret float %ext
}

define float @rsqrtss(float* %a) {
; CHECK-LABEL: rsqrtss:
; CHECK:       vrsqrtss (%rdi), %xmm0, %xmm0

    %ld = load float, float* %a
    %ins = insertelement <4 x float> undef, float %ld, i32 0
    %res = tail call <4 x float> @llvm.x86.sse.rsqrt.ss(<4 x float> %ins)
    %ext = extractelement <4 x float> %res, i32 0
    ret float %ext
}

define float @sqrtss(float* %a) {
; CHECK-LABEL: sqrtss:
; CHECK:       vsqrtss (%rdi), %xmm0, %xmm0

    %ld = load float, float* %a
    %ins = insertelement <4 x float> undef, float %ld, i32 0
    %res = tail call <4 x float> @llvm.x86.sse.sqrt.ss(<4 x float> %ins)
    %ext = extractelement <4 x float> %res, i32 0
    ret float %ext
}

define double @sqrtsd(double* %a) {
; CHECK-LABEL: sqrtsd:
; CHECK:       vsqrtsd (%rdi), %xmm0, %xmm0

    %ld = load double, double* %a
    %ins = insertelement <2 x double> undef, double %ld, i32 0
    %res = tail call <2 x double> @llvm.x86.sse2.sqrt.sd(<2 x double> %ins)
    %ext = extractelement <2 x double> %res, i32 0
    ret double %ext
}


declare <4 x float> @llvm.x86.sse.rcp.ss(<4 x float>) nounwind readnone
declare <4 x float> @llvm.x86.sse.rsqrt.ss(<4 x float>) nounwind readnone
declare <4 x float> @llvm.x86.sse.sqrt.ss(<4 x float>) nounwind readnone
declare <2 x double> @llvm.x86.sse2.sqrt.sd(<2 x double>) nounwind readnone

