; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=sse2 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=avx,use-sqrt-est | FileCheck %s --check-prefix=ESTIMATE

declare double @__sqrt_finite(double) #0
declare float @__sqrtf_finite(float) #0
declare x86_fp80 @__sqrtl_finite(x86_fp80) #0
declare float @llvm.sqrt.f32(float) #0
declare <4 x float> @llvm.sqrt.v4f32(<4 x float>) #0
declare <8 x float> @llvm.sqrt.v8f32(<8 x float>) #0


define double @fd(double %d) #0 {
; CHECK-LABEL: fd:
; CHECK:       # BB#0:
; CHECK-NEXT:    sqrtsd %xmm0, %xmm0
; CHECK-NEXT:    retq
;
; ESTIMATE-LABEL: fd:
; ESTIMATE:       # BB#0:
; ESTIMATE-NEXT:    vsqrtsd %xmm0, %xmm0, %xmm0
; ESTIMATE-NEXT:    retq
  %call = tail call double @__sqrt_finite(double %d) #1
  ret double %call
}


define float @ff(float %f) #0 {
; CHECK-LABEL: ff:
; CHECK:       # BB#0:
; CHECK-NEXT:    sqrtss %xmm0, %xmm0
; CHECK-NEXT:    retq
;
; ESTIMATE-LABEL: ff:
; ESTIMATE:       # BB#0:
; ESTIMATE-NEXT:    vrsqrtss %xmm0, %xmm0, %xmm1
; ESTIMATE-NEXT:    vmulss {{.*}}(%rip), %xmm1, %xmm2
; ESTIMATE-NEXT:    vmulss %xmm1, %xmm1, %xmm1
; ESTIMATE-NEXT:    vmulss %xmm0, %xmm1, %xmm1
; ESTIMATE-NEXT:    vaddss {{.*}}(%rip), %xmm1, %xmm1
; ESTIMATE-NEXT:    vmulss %xmm2, %xmm1, %xmm1
; ESTIMATE-NEXT:    vmulss %xmm1, %xmm0, %xmm1
; ESTIMATE-NEXT:    vxorps %xmm2, %xmm2, %xmm2
; ESTIMATE-NEXT:    vcmpeqss %xmm2, %xmm0, %xmm0
; ESTIMATE-NEXT:    vandnps %xmm1, %xmm0, %xmm0
; ESTIMATE-NEXT:    retq
  %call = tail call float @__sqrtf_finite(float %f) #1
  ret float %call
}


define x86_fp80 @fld(x86_fp80 %ld) #0 {
; CHECK-LABEL: fld:
; CHECK:       # BB#0:
; CHECK-NEXT:    fldt {{[0-9]+}}(%rsp)
; CHECK-NEXT:    fsqrt
; CHECK-NEXT:    retq
;
; ESTIMATE-LABEL: fld:
; ESTIMATE:       # BB#0:
; ESTIMATE-NEXT:    fldt {{[0-9]+}}(%rsp)
; ESTIMATE-NEXT:    fsqrt
; ESTIMATE-NEXT:    retq
  %call = tail call x86_fp80 @__sqrtl_finite(x86_fp80 %ld) #1
  ret x86_fp80 %call
}



define float @reciprocal_square_root(float %x) #0 {
; CHECK-LABEL: reciprocal_square_root:
; CHECK:       # BB#0:
; CHECK-NEXT:    sqrtss %xmm0, %xmm1
; CHECK-NEXT:    movss {{.*#+}} xmm0 = mem[0],zero,zero,zero
; CHECK-NEXT:    divss %xmm1, %xmm0
; CHECK-NEXT:    retq
;
; ESTIMATE-LABEL: reciprocal_square_root:
; ESTIMATE:       # BB#0:
; ESTIMATE-NEXT:    vrsqrtss %xmm0, %xmm0, %xmm1
; ESTIMATE-NEXT:    vmulss {{.*}}(%rip), %xmm1, %xmm2
; ESTIMATE-NEXT:    vmulss %xmm1, %xmm1, %xmm1
; ESTIMATE-NEXT:    vmulss %xmm0, %xmm1, %xmm0
; ESTIMATE-NEXT:    vaddss {{.*}}(%rip), %xmm0, %xmm0
; ESTIMATE-NEXT:    vmulss %xmm2, %xmm0, %xmm0
; ESTIMATE-NEXT:    retq
  %sqrt = tail call float @llvm.sqrt.f32(float %x)
  %div = fdiv fast float 1.0, %sqrt
  ret float %div
}

define <4 x float> @reciprocal_square_root_v4f32(<4 x float> %x) #0 {
; CHECK-LABEL: reciprocal_square_root_v4f32:
; CHECK:       # BB#0:
; CHECK-NEXT:    sqrtps %xmm0, %xmm1
; CHECK-NEXT:    movaps {{.*#+}} xmm0 = [1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00]
; CHECK-NEXT:    divps %xmm1, %xmm0
; CHECK-NEXT:    retq
;
; ESTIMATE-LABEL: reciprocal_square_root_v4f32:
; ESTIMATE:       # BB#0:
; ESTIMATE-NEXT:    vrsqrtps %xmm0, %xmm1
; ESTIMATE-NEXT:    vmulps %xmm1, %xmm1, %xmm2
; ESTIMATE-NEXT:    vmulps %xmm0, %xmm2, %xmm0
; ESTIMATE-NEXT:    vaddps {{.*}}(%rip), %xmm0, %xmm0
; ESTIMATE-NEXT:    vmulps {{.*}}(%rip), %xmm1, %xmm1
; ESTIMATE-NEXT:    vmulps %xmm1, %xmm0, %xmm0
; ESTIMATE-NEXT:    retq
  %sqrt = tail call <4 x float> @llvm.sqrt.v4f32(<4 x float> %x)
  %div = fdiv fast <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, %sqrt
  ret <4 x float> %div
}

define <8 x float> @reciprocal_square_root_v8f32(<8 x float> %x) #0 {
; CHECK-LABEL: reciprocal_square_root_v8f32:
; CHECK:       # BB#0:
; CHECK-NEXT:    sqrtps %xmm1, %xmm2
; CHECK-NEXT:    sqrtps %xmm0, %xmm3
; CHECK-NEXT:    movaps {{.*#+}} xmm1 = [1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00]
; CHECK-NEXT:    movaps %xmm1, %xmm0
; CHECK-NEXT:    divps %xmm3, %xmm0
; CHECK-NEXT:    divps %xmm2, %xmm1
; CHECK-NEXT:    retq
;
; ESTIMATE-LABEL: reciprocal_square_root_v8f32:
; ESTIMATE:       # BB#0:
; ESTIMATE-NEXT:    vrsqrtps %ymm0, %ymm1
; ESTIMATE-NEXT:    vmulps %ymm1, %ymm1, %ymm2
; ESTIMATE-NEXT:    vmulps %ymm0, %ymm2, %ymm0
; ESTIMATE-NEXT:    vaddps {{.*}}(%rip), %ymm0, %ymm0
; ESTIMATE-NEXT:    vmulps {{.*}}(%rip), %ymm1, %ymm1
; ESTIMATE-NEXT:    vmulps %ymm1, %ymm0, %ymm0
; ESTIMATE-NEXT:    retq
  %sqrt = tail call <8 x float> @llvm.sqrt.v8f32(<8 x float> %x)
  %div = fdiv fast <8 x float> <float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0>, %sqrt
  ret <8 x float> %div
}


attributes #0 = { "unsafe-fp-math"="true" }
attributes #1 = { nounwind readnone }

