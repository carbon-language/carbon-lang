; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl --show-mc-encoding| FileCheck %s

; CHECK-LABEL: addpd512
; CHECK: vaddpd
; CHECK: ret
define <8 x double> @addpd512(<8 x double> %y, <8 x double> %x) {
entry:
  %add.i = fadd <8 x double> %x, %y
  ret <8 x double> %add.i
}

; CHECK-LABEL: addpd512fold
; CHECK: vaddpd LCP{{.*}}(%rip)
; CHECK: ret
define <8 x double> @addpd512fold(<8 x double> %y) {
entry:
  %add.i = fadd <8 x double> %y, <double 4.500000e+00, double 3.400000e+00, double 2.300000e+00, double 1.200000e+00, double 4.500000e+00, double 3.800000e+00, double 2.300000e+00, double 1.200000e+00>
  ret <8 x double> %add.i
}

; CHECK-LABEL: addps512
; CHECK: vaddps
; CHECK: ret
define <16 x float> @addps512(<16 x float> %y, <16 x float> %x) {
entry:
  %add.i = fadd <16 x float> %x, %y
  ret <16 x float> %add.i
}

; CHECK-LABEL: addps512fold
; CHECK: vaddps LCP{{.*}}(%rip)
; CHECK: ret
define <16 x float> @addps512fold(<16 x float> %y) {
entry:
  %add.i = fadd <16 x float> %y, <float 4.500000e+00, float 0x400B333340000000, float 0x4002666660000000, float 0x3FF3333340000000, float 4.500000e+00, float 0x400B333340000000, float 0x4002666660000000, float 0x3FF3333340000000, float 4.500000e+00, float 0x400B333340000000, float 0x4002666660000000, float 4.500000e+00, float 4.500000e+00, float 0x400B333340000000,  float 0x4002666660000000, float 0x3FF3333340000000>
  ret <16 x float> %add.i
}

; CHECK-LABEL: subpd512
; CHECK: vsubpd
; CHECK: ret
define <8 x double> @subpd512(<8 x double> %y, <8 x double> %x) {
entry:
  %sub.i = fsub <8 x double> %x, %y
  ret <8 x double> %sub.i
}

; CHECK-LABEL: @subpd512fold
; CHECK: vsubpd (%
; CHECK: ret
define <8 x double> @subpd512fold(<8 x double> %y, <8 x double>* %x) {
entry:
  %tmp2 = load <8 x double>* %x, align 8
  %sub.i = fsub <8 x double> %y, %tmp2
  ret <8 x double> %sub.i
}

; CHECK-LABEL: @subps512
; CHECK: vsubps
; CHECK: ret
define <16 x float> @subps512(<16 x float> %y, <16 x float> %x) {
entry:
  %sub.i = fsub <16 x float> %x, %y
  ret <16 x float> %sub.i
}

; CHECK-LABEL: subps512fold
; CHECK: vsubps (%
; CHECK: ret
define <16 x float> @subps512fold(<16 x float> %y, <16 x float>* %x) {
entry:
  %tmp2 = load <16 x float>* %x, align 4
  %sub.i = fsub <16 x float> %y, %tmp2
  ret <16 x float> %sub.i
}

; CHECK-LABEL: imulq512
; CHECK: vpmuludq
; CHECK: vpmuludq
; CHECK: ret
define <8 x i64> @imulq512(<8 x i64> %y, <8 x i64> %x) {
  %z = mul <8 x i64>%x, %y
  ret <8 x i64>%z
}

; CHECK-LABEL: mulpd512
; CHECK: vmulpd
; CHECK: ret
define <8 x double> @mulpd512(<8 x double> %y, <8 x double> %x) {
entry:
  %mul.i = fmul <8 x double> %x, %y
  ret <8 x double> %mul.i
}

; CHECK-LABEL: mulpd512fold
; CHECK: vmulpd LCP{{.*}}(%rip)
; CHECK: ret
define <8 x double> @mulpd512fold(<8 x double> %y) {
entry:
  %mul.i = fmul <8 x double> %y, <double 4.500000e+00, double 3.400000e+00, double 2.300000e+00, double 1.200000e+00, double 4.500000e+00, double 3.400000e+00, double 2.300000e+00, double 1.200000e+00>
  ret <8 x double> %mul.i
}

; CHECK-LABEL: mulps512
; CHECK: vmulps
; CHECK: ret
define <16 x float> @mulps512(<16 x float> %y, <16 x float> %x) {
entry:
  %mul.i = fmul <16 x float> %x, %y
  ret <16 x float> %mul.i
}

; CHECK-LABEL: mulps512fold
; CHECK: vmulps LCP{{.*}}(%rip)
; CHECK: ret
define <16 x float> @mulps512fold(<16 x float> %y) {
entry:
  %mul.i = fmul <16 x float> %y, <float 4.500000e+00, float 0x400B333340000000, float 0x4002666660000000, float 0x3FF3333340000000, float 4.500000e+00, float 0x400B333340000000, float 0x4002666660000000, float 0x3FF3333340000000, float 4.500000e+00, float 0x400B333340000000, float 0x4002666660000000, float 0x3FF3333340000000, float 4.500000e+00, float 0x400B333340000000, float 0x4002666660000000, float 0x3FF3333340000000>
  ret <16 x float> %mul.i
}

; CHECK-LABEL: divpd512
; CHECK: vdivpd
; CHECK: ret
define <8 x double> @divpd512(<8 x double> %y, <8 x double> %x) {
entry:
  %div.i = fdiv <8 x double> %x, %y
  ret <8 x double> %div.i
}

; CHECK-LABEL: divpd512fold
; CHECK: vdivpd LCP{{.*}}(%rip)
; CHECK: ret
define <8 x double> @divpd512fold(<8 x double> %y) {
entry:
  %div.i = fdiv <8 x double> %y, <double 4.500000e+00, double 3.400000e+00, double 2.300000e+00, double 1.200000e+00, double 4.500000e+00, double 3.400000e+00, double 2.300000e+00, double 1.200000e+00>
  ret <8 x double> %div.i
}

; CHECK-LABEL: divps512
; CHECK: vdivps
; CHECK: ret
define <16 x float> @divps512(<16 x float> %y, <16 x float> %x) {
entry:
  %div.i = fdiv <16 x float> %x, %y
  ret <16 x float> %div.i
}

; CHECK-LABEL: divps512fold
; CHECK: vdivps LCP{{.*}}(%rip)
; CHECK: ret
define <16 x float> @divps512fold(<16 x float> %y) {
entry:
  %div.i = fdiv <16 x float> %y, <float 4.500000e+00, float 0x400B333340000000, float 0x4002666660000000, float 0x3FF3333340000000, float 4.500000e+00, float 4.500000e+00, float 0x4002666660000000, float 0x3FF3333340000000, float 4.500000e+00, float 0x400B333340000000, float 0x4002666660000000, float 0x3FF3333340000000, float 4.500000e+00, float 4.500000e+00, float 0x4002666660000000, float 0x3FF3333340000000>
  ret <16 x float> %div.i
}

; CHECK-LABEL: vpaddq_test
; CHECK: vpaddq %zmm
; CHECK: ret
define <8 x i64> @vpaddq_test(<8 x i64> %i, <8 x i64> %j) nounwind readnone {
  %x = add <8 x i64> %i, %j
  ret <8 x i64> %x
}

; CHECK-LABEL: vpaddd_test
; CHECK: vpaddd %zmm
; CHECK: ret
define <16 x i32> @vpaddd_test(<16 x i32> %i, <16 x i32> %j) nounwind readnone {
  %x = add <16 x i32> %i, %j
  ret <16 x i32> %x
}

; CHECK-LABEL: vpsubq_test
; CHECK: vpsubq %zmm
; CHECK: ret
define <8 x i64> @vpsubq_test(<8 x i64> %i, <8 x i64> %j) nounwind readnone {
  %x = sub <8 x i64> %i, %j
  ret <8 x i64> %x
}

; CHECK-LABEL: vpsubd_test
; CHECK: vpsubd
; CHECK: ret
define <16 x i32> @vpsubd_test(<16 x i32> %i, <16 x i32> %j) nounwind readnone {
  %x = sub <16 x i32> %i, %j
  ret <16 x i32> %x
}

; CHECK-LABEL: vpmulld_test
; CHECK: vpmulld %zmm
; CHECK: ret
define <16 x i32> @vpmulld_test(<16 x i32> %i, <16 x i32> %j) {
  %x = mul <16 x i32> %i, %j
  ret <16 x i32> %x
}

; CHECK-LABEL: sqrtA
; CHECK: vsqrtss {{.*}} encoding: [0x62
; CHECK: ret
declare float @sqrtf(float) readnone
define float @sqrtA(float %a) nounwind uwtable readnone ssp {
entry:
  %conv1 = tail call float @sqrtf(float %a) nounwind readnone
  ret float %conv1
}

; CHECK-LABEL: sqrtB
; CHECK: vsqrtsd {{.*}}## encoding: [0x62
; CHECK: ret
declare double @sqrt(double) readnone
define double @sqrtB(double %a) nounwind uwtable readnone ssp {
entry:
  %call = tail call double @sqrt(double %a) nounwind readnone
  ret double %call
}

; CHECK-LABEL: sqrtC
; CHECK: vsqrtss {{.*}}## encoding: [0x62
; CHECK: ret
declare float @llvm.sqrt.f32(float)
define float @sqrtC(float %a) nounwind {
  %b = call float @llvm.sqrt.f32(float %a)
  ret float %b
}

; CHECK-LABEL: sqrtD
; CHECK: vsqrtps {{.*}}
; CHECK: ret
declare <16 x float> @llvm.sqrt.v16f32(<16 x float>)
define <16 x float> @sqrtD(<16 x float> %a) nounwind {
  %b = call <16 x float> @llvm.sqrt.v16f32(<16 x float> %a)
  ret <16 x float> %b
}

; CHECK-LABEL: sqrtE
; CHECK: vsqrtpd {{.*}}
; CHECK: ret
declare <8 x double> @llvm.sqrt.v8f64(<8 x double>)
define <8 x double> @sqrtE(<8 x double> %a) nounwind {
  %b = call <8 x double> @llvm.sqrt.v8f64(<8 x double> %a)
  ret <8 x double> %b
}

; CHECK-LABEL: fadd_broadcast
; CHECK: LCP{{.*}}(%rip){1to16}, %zmm0, %zmm0
; CHECK: ret
define <16 x float> @fadd_broadcast(<16 x float> %a) nounwind {
  %b = fadd <16 x float> %a, <float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000>
  ret <16 x float> %b
}

; CHECK-LABEL: addq_broadcast
; CHECK: vpaddq LCP{{.*}}(%rip){1to8}, %zmm0, %zmm0
; CHECK: ret
define <8 x i64> @addq_broadcast(<8 x i64> %a) nounwind {
  %b = add <8 x i64> %a, <i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2>
  ret <8 x i64> %b
}

; CHECK-LABEL: orq_broadcast
; CHECK: vporq LCP{{.*}}(%rip){1to8}, %zmm0, %zmm0
; CHECK: ret
define <8 x i64> @orq_broadcast(<8 x i64> %a) nounwind {
  %b = or <8 x i64> %a, <i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2>
  ret <8 x i64> %b
}

; CHECK-LABEL: andd512fold
; CHECK: vpandd (%
; CHECK: ret
define <16 x i32> @andd512fold(<16 x i32> %y, <16 x i32>* %x) {
entry:
  %a = load <16 x i32>* %x, align 4
  %b = and <16 x i32> %y, %a
  ret <16 x i32> %b
}

; CHECK-LABEL: andqbrst
; CHECK: vpandq  (%rdi){1to8}, %zmm
; CHECK: ret
define <8 x i64> @andqbrst(<8 x i64> %p1, i64* %ap) {
entry:
  %a = load i64* %ap, align 8
  %b = insertelement <8 x i64> undef, i64 %a, i32 0
  %c = shufflevector <8 x i64> %b, <8 x i64> undef, <8 x i32> zeroinitializer
  %d = and <8 x i64> %p1, %c
  ret <8 x i64>%d
}
