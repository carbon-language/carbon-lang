; RUN: llc -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=sse -enable-unsafe-fp-math < %s | FileCheck %s --check-prefix=SSE
; RUN: llc -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=avx -enable-unsafe-fp-math < %s | FileCheck %s --check-prefix=AVX

; Verify that the first two adds are independent regardless of how the inputs are
; commuted. The destination registers are used as source registers for the third add.

define float @reassociate_adds1(float %x0, float %x1, float %x2, float %x3) {
; SSE-LABEL: reassociate_adds1:
; SSE:       # BB#0:
; SSE-NEXT:    addss %xmm1, %xmm0
; SSE-NEXT:    addss %xmm3, %xmm2
; SSE-NEXT:    addss %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: reassociate_adds1:
; AVX:       # BB#0:
; AVX-NEXT:    vaddss %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vaddss %xmm3, %xmm2, %xmm1
; AVX-NEXT:    vaddss %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
  %t0 = fadd float %x0, %x1
  %t1 = fadd float %t0, %x2
  %t2 = fadd float %t1, %x3
  ret float %t2
}

define float @reassociate_adds2(float %x0, float %x1, float %x2, float %x3) {
; SSE-LABEL: reassociate_adds2:
; SSE:       # BB#0:
; SSE-NEXT:    addss %xmm1, %xmm0
; SSE-NEXT:    addss %xmm3, %xmm2
; SSE-NEXT:    addss %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: reassociate_adds2:
; AVX:       # BB#0:
; AVX-NEXT:    vaddss %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vaddss %xmm3, %xmm2, %xmm1
; AVX-NEXT:    vaddss %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
  %t0 = fadd float %x0, %x1
  %t1 = fadd float %x2, %t0
  %t2 = fadd float %t1, %x3
  ret float %t2
}

define float @reassociate_adds3(float %x0, float %x1, float %x2, float %x3) {
; SSE-LABEL: reassociate_adds3:
; SSE:       # BB#0:
; SSE-NEXT:    addss %xmm1, %xmm0
; SSE-NEXT:    addss %xmm3, %xmm2
; SSE-NEXT:    addss %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: reassociate_adds3:
; AVX:       # BB#0:
; AVX-NEXT:    vaddss %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vaddss %xmm3, %xmm2, %xmm1
; AVX-NEXT:    vaddss %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
  %t0 = fadd float %x0, %x1
  %t1 = fadd float %t0, %x2
  %t2 = fadd float %x3, %t1
  ret float %t2
}

define float @reassociate_adds4(float %x0, float %x1, float %x2, float %x3) {
; SSE-LABEL: reassociate_adds4:
; SSE:       # BB#0:
; SSE-NEXT:    addss %xmm1, %xmm0
; SSE-NEXT:    addss %xmm3, %xmm2
; SSE-NEXT:    addss %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: reassociate_adds4:
; AVX:       # BB#0:
; AVX-NEXT:    vaddss %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vaddss %xmm3, %xmm2, %xmm1
; AVX-NEXT:    vaddss %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
  %t0 = fadd float %x0, %x1
  %t1 = fadd float %x2, %t0
  %t2 = fadd float %x3, %t1
  ret float %t2
}

; Verify that we reassociate some of these ops. The optimal balanced tree of adds is not
; produced because that would cost more compile time.

define float @reassociate_adds5(float %x0, float %x1, float %x2, float %x3, float %x4, float %x5, float %x6, float %x7) {
; SSE-LABEL: reassociate_adds5:
; SSE:       # BB#0:
; SSE-NEXT:    addss %xmm1, %xmm0
; SSE-NEXT:    addss %xmm3, %xmm2
; SSE-NEXT:    addss %xmm2, %xmm0
; SSE-NEXT:    addss %xmm5, %xmm4
; SSE-NEXT:    addss %xmm6, %xmm4
; SSE-NEXT:    addss %xmm4, %xmm0
; SSE-NEXT:    addss %xmm7, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: reassociate_adds5:
; AVX:       # BB#0:
; AVX-NEXT:    vaddss %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vaddss %xmm3, %xmm2, %xmm1
; AVX-NEXT:    vaddss %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vaddss %xmm5, %xmm4, %xmm1
; AVX-NEXT:    vaddss %xmm6, %xmm1, %xmm1
; AVX-NEXT:    vaddss %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vaddss %xmm7, %xmm0, %xmm0
; AVX-NEXT:    retq
  %t0 = fadd float %x0, %x1
  %t1 = fadd float %t0, %x2
  %t2 = fadd float %t1, %x3
  %t3 = fadd float %t2, %x4
  %t4 = fadd float %t3, %x5
  %t5 = fadd float %t4, %x6
  %t6 = fadd float %t5, %x7
  ret float %t6
}

; Verify that we only need two associative operations to reassociate the operands.
; Also, we should reassociate such that the result of the high latency division
; is used by the final 'add' rather than reassociating the %x3 operand with the
; division. The latter reassociation would not improve anything.

define float @reassociate_adds6(float %x0, float %x1, float %x2, float %x3) {
; SSE-LABEL: reassociate_adds6:
; SSE:       # BB#0:
; SSE-NEXT:    divss %xmm1, %xmm0
; SSE-NEXT:    addss %xmm3, %xmm2
; SSE-NEXT:    addss %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: reassociate_adds6:
; AVX:       # BB#0:
; AVX-NEXT:    vdivss %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vaddss %xmm3, %xmm2, %xmm1
; AVX-NEXT:    vaddss %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
  %t0 = fdiv float %x0, %x1
  %t1 = fadd float %x2, %t0
  %t2 = fadd float %x3, %t1
  ret float %t2
}

; Verify that SSE and AVX scalar single-precision multiplies are reassociated.

define float @reassociate_muls1(float %x0, float %x1, float %x2, float %x3) {
; SSE-LABEL: reassociate_muls1:
; SSE:       # BB#0:
; SSE-NEXT:    divss %xmm1, %xmm0
; SSE-NEXT:    mulss %xmm3, %xmm2
; SSE-NEXT:    mulss %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: reassociate_muls1:
; AVX:       # BB#0:
; AVX-NEXT:    vdivss %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vmulss %xmm3, %xmm2, %xmm1
; AVX-NEXT:    vmulss %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
  %t0 = fdiv float %x0, %x1
  %t1 = fmul float %x2, %t0
  %t2 = fmul float %x3, %t1
  ret float %t2
}

; Verify that SSE and AVX scalar double-precision adds are reassociated.

define double @reassociate_adds_double(double %x0, double %x1, double %x2, double %x3) {
; SSE-LABEL: reassociate_adds_double:
; SSE:       # BB#0:
; SSE-NEXT:    divsd %xmm1, %xmm0
; SSE-NEXT:    addsd %xmm3, %xmm2
; SSE-NEXT:    addsd %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: reassociate_adds_double:
; AVX:       # BB#0:
; AVX-NEXT:    vdivsd %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vaddsd %xmm3, %xmm2, %xmm1
; AVX-NEXT:    vaddsd %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
  %t0 = fdiv double %x0, %x1
  %t1 = fadd double %x2, %t0
  %t2 = fadd double %x3, %t1
  ret double %t2
}

; Verify that SSE and AVX scalar double-precision multiplies are reassociated.

define double @reassociate_muls_double(double %x0, double %x1, double %x2, double %x3) {
; SSE-LABEL: reassociate_muls_double:
; SSE:       # BB#0:
; SSE-NEXT:    divsd %xmm1, %xmm0
; SSE-NEXT:    mulsd %xmm3, %xmm2
; SSE-NEXT:    mulsd %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: reassociate_muls_double:
; AVX:       # BB#0:
; AVX-NEXT:    vdivsd %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vmulsd %xmm3, %xmm2, %xmm1
; AVX-NEXT:    vmulsd %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
  %t0 = fdiv double %x0, %x1
  %t1 = fmul double %x2, %t0
  %t2 = fmul double %x3, %t1
  ret double %t2
}

; Verify that SSE and AVX 128-bit vector single-precision adds are reassociated.

define <4 x float> @reassociate_adds_v4f32(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, <4 x float> %x3) {
; SSE-LABEL: reassociate_adds_v4f32:
; SSE:       # BB#0:
; SSE-NEXT:    mulps %xmm1, %xmm0
; SSE-NEXT:    addps %xmm3, %xmm2
; SSE-NEXT:    addps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: reassociate_adds_v4f32:
; AVX:       # BB#0:
; AVX-NEXT:    vmulps %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vaddps %xmm3, %xmm2, %xmm1
; AVX-NEXT:    vaddps %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
  %t0 = fmul <4 x float> %x0, %x1
  %t1 = fadd <4 x float> %x2, %t0
  %t2 = fadd <4 x float> %x3, %t1
  ret <4 x float> %t2
}

; Verify that SSE and AVX 128-bit vector double-precision adds are reassociated.

define <2 x double> @reassociate_adds_v2f64(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, <2 x double> %x3) {
; SSE-LABEL: reassociate_adds_v2f64:
; SSE:       # BB#0:
; SSE-NEXT:    mulpd %xmm1, %xmm0
; SSE-NEXT:    addpd %xmm3, %xmm2
; SSE-NEXT:    addpd %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: reassociate_adds_v2f64:
; AVX:       # BB#0:
; AVX-NEXT:    vmulpd %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vaddpd %xmm3, %xmm2, %xmm1
; AVX-NEXT:    vaddpd %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
  %t0 = fmul <2 x double> %x0, %x1
  %t1 = fadd <2 x double> %x2, %t0
  %t2 = fadd <2 x double> %x3, %t1
  ret <2 x double> %t2
}

; Verify that SSE and AVX 128-bit vector single-precision multiplies are reassociated.

define <4 x float> @reassociate_muls_v4f32(<4 x float> %x0, <4 x float> %x1, <4 x float> %x2, <4 x float> %x3) {
; SSE-LABEL: reassociate_muls_v4f32:
; SSE:       # BB#0:
; SSE-NEXT:    addps %xmm1, %xmm0
; SSE-NEXT:    mulps %xmm3, %xmm2
; SSE-NEXT:    mulps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: reassociate_muls_v4f32:
; AVX:       # BB#0:
; AVX-NEXT:    vaddps %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vmulps %xmm3, %xmm2, %xmm1
; AVX-NEXT:    vmulps %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
  %t0 = fadd <4 x float> %x0, %x1
  %t1 = fmul <4 x float> %x2, %t0
  %t2 = fmul <4 x float> %x3, %t1
  ret <4 x float> %t2
}

; Verify that SSE and AVX 128-bit vector double-precision multiplies are reassociated.

define <2 x double> @reassociate_muls_v2f64(<2 x double> %x0, <2 x double> %x1, <2 x double> %x2, <2 x double> %x3) {
; SSE-LABEL: reassociate_muls_v2f64:
; SSE:       # BB#0:
; SSE-NEXT:    addpd %xmm1, %xmm0
; SSE-NEXT:    mulpd %xmm3, %xmm2
; SSE-NEXT:    mulpd %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: reassociate_muls_v2f64:
; AVX:       # BB#0:
; AVX-NEXT:    vaddpd %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vmulpd %xmm3, %xmm2, %xmm1
; AVX-NEXT:    vmulpd %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
  %t0 = fadd <2 x double> %x0, %x1
  %t1 = fmul <2 x double> %x2, %t0
  %t2 = fmul <2 x double> %x3, %t1
  ret <2 x double> %t2
}

; Verify that AVX 256-bit vector single-precision adds are reassociated.

define <8 x float> @reassociate_adds_v8f32(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, <8 x float> %x3) {
; AVX-LABEL: reassociate_adds_v8f32:
; AVX:       # BB#0:
; AVX-NEXT:    vmulps %ymm1, %ymm0, %ymm0
; AVX-NEXT:    vaddps %ymm3, %ymm2, %ymm1
; AVX-NEXT:    vaddps %ymm1, %ymm0, %ymm0
; AVX-NEXT:    retq
  %t0 = fmul <8 x float> %x0, %x1
  %t1 = fadd <8 x float> %x2, %t0
  %t2 = fadd <8 x float> %x3, %t1
  ret <8 x float> %t2
}

; Verify that AVX 256-bit vector double-precision adds are reassociated.

define <4 x double> @reassociate_adds_v4f64(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, <4 x double> %x3) {
; AVX-LABEL: reassociate_adds_v4f64:
; AVX:       # BB#0:
; AVX-NEXT:    vmulpd %ymm1, %ymm0, %ymm0
; AVX-NEXT:    vaddpd %ymm3, %ymm2, %ymm1
; AVX-NEXT:    vaddpd %ymm1, %ymm0, %ymm0
; AVX-NEXT:    retq
  %t0 = fmul <4 x double> %x0, %x1
  %t1 = fadd <4 x double> %x2, %t0
  %t2 = fadd <4 x double> %x3, %t1
  ret <4 x double> %t2
}

; Verify that AVX 256-bit vector single-precision multiplies are reassociated.

define <8 x float> @reassociate_muls_v8f32(<8 x float> %x0, <8 x float> %x1, <8 x float> %x2, <8 x float> %x3) {
; AVX-LABEL: reassociate_muls_v8f32:
; AVX:       # BB#0:
; AVX-NEXT:    vaddps %ymm1, %ymm0, %ymm0
; AVX-NEXT:    vmulps %ymm3, %ymm2, %ymm1
; AVX-NEXT:    vmulps %ymm1, %ymm0, %ymm0
; AVX-NEXT:    retq
  %t0 = fadd <8 x float> %x0, %x1
  %t1 = fmul <8 x float> %x2, %t0
  %t2 = fmul <8 x float> %x3, %t1
  ret <8 x float> %t2
}

; Verify that AVX 256-bit vector double-precision multiplies are reassociated.

define <4 x double> @reassociate_muls_v4f64(<4 x double> %x0, <4 x double> %x1, <4 x double> %x2, <4 x double> %x3) {
; AVX-LABEL: reassociate_muls_v4f64:
; AVX:       # BB#0:
; AVX-NEXT:    vaddpd %ymm1, %ymm0, %ymm0
; AVX-NEXT:    vmulpd %ymm3, %ymm2, %ymm1
; AVX-NEXT:    vmulpd %ymm1, %ymm0, %ymm0
; AVX-NEXT:    retq
  %t0 = fadd <4 x double> %x0, %x1
  %t1 = fmul <4 x double> %x2, %t0
  %t2 = fmul <4 x double> %x3, %t1
  ret <4 x double> %t2
}

; Verify that SSE and AVX scalar single-precision minimum ops are reassociated.

define float @reassociate_mins_single(float %x0, float %x1, float %x2, float %x3) {
; SSE-LABEL: reassociate_mins_single:
; SSE:       # BB#0:
; SSE-NEXT:    divss %xmm1, %xmm0
; SSE-NEXT:    minss %xmm3, %xmm2
; SSE-NEXT:    minss %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: reassociate_mins_single:
; AVX:       # BB#0:
; AVX-NEXT:    vdivss %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vminss %xmm3, %xmm2, %xmm1
; AVX-NEXT:    vminss %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
  %t0 = fdiv float %x0, %x1
  %cmp1 = fcmp olt float %x2, %t0
  %sel1 = select i1 %cmp1, float %x2, float %t0
  %cmp2 = fcmp olt float %x3, %sel1
  %sel2 = select i1 %cmp2, float %x3, float %sel1
  ret float %sel2
}

