; RUN: llc < %s -mtriple armv8 -mattr=+neon,+fp-armv8 | FileCheck %s
; RUN: llc < %s -mtriple armv8 -mattr=+neon,+fp-armv8 \
; RUN:          -enable-no-nans-fp-math -enable-unsafe-fp-math | FileCheck %s --check-prefix=CHECK-FAST

; vectors

define <4 x float> @vmaxnmq(<4 x float>* %A, <4 x float>* %B) nounwind {
; CHECK-LABEL: vmaxnmq:
; CHECK: vmaxnm.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
  %tmp1 = load <4 x float>, <4 x float>* %A
  %tmp2 = load <4 x float>, <4 x float>* %B
  %tmp3 = call <4 x float> @llvm.arm.neon.vmaxnm.v4f32(<4 x float> %tmp1, <4 x float> %tmp2)
  ret <4 x float> %tmp3
}

define <2 x float> @vmaxnmd(<2 x float>* %A, <2 x float>* %B) nounwind {
; CHECK-LABEL: vmaxnmd:
; CHECK: vmaxnm.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
  %tmp1 = load <2 x float>, <2 x float>* %A
  %tmp2 = load <2 x float>, <2 x float>* %B
  %tmp3 = call <2 x float> @llvm.arm.neon.vmaxnm.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
  ret <2 x float> %tmp3
}

define <4 x float> @vminnmq(<4 x float>* %A, <4 x float>* %B) nounwind {
; CHECK-LABEL: vminnmq:
; CHECK: vminnm.f32 q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
  %tmp1 = load <4 x float>, <4 x float>* %A
  %tmp2 = load <4 x float>, <4 x float>* %B
  %tmp3 = call <4 x float> @llvm.arm.neon.vminnm.v4f32(<4 x float> %tmp1, <4 x float> %tmp2)
  ret <4 x float> %tmp3
}

define <2 x float> @vminnmd(<2 x float>* %A, <2 x float>* %B) nounwind {
; CHECK-LABEL: vminnmd:
; CHECK: vminnm.f32 d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
  %tmp1 = load <2 x float>, <2 x float>* %A
  %tmp2 = load <2 x float>, <2 x float>* %B
  %tmp3 = call <2 x float> @llvm.arm.neon.vminnm.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
  ret <2 x float> %tmp3
}

; scalars

define float @fp-armv8_vminnm_o(float %a, float %b) {
; CHECK-FAST-LABEL: "fp-armv8_vminnm_o":
; CHECK-FAST-NOT: vcmp
; CHECK-FAST: vminnm.f32
; CHECK-LABEL: "fp-armv8_vminnm_o":
; CHECK-NOT: vminnm.f32
  %cmp = fcmp olt float %a, %b
  %cond = select i1 %cmp, float %a, float %b
  ret float %cond
}

define double @fp-armv8_vminnm_ole(double %a, double %b) {
; CHECK-FAST-LABEL: "fp-armv8_vminnm_ole":
; CHECK-FAST-NOT: vcmp
; CHECK-FAST: vminnm.f64
; CHECK-LABEL: "fp-armv8_vminnm_ole":
; CHECK-NOT: vminnm.f64
  %cmp = fcmp ole double %a, %b
  %cond = select i1 %cmp, double %a, double %b
  ret double %cond
}

define float @fp-armv8_vminnm_o_rev(float %a, float %b) {
; CHECK-FAST-LABEL: "fp-armv8_vminnm_o_rev":
; CHECK-FAST-NOT: vcmp
; CHECK-FAST: vminnm.f32
; CHECK-LABEL: "fp-armv8_vminnm_o_rev":
; CHECK-NOT: vminnm.f32
  %cmp = fcmp ogt float %a, %b
  %cond = select i1 %cmp, float %b, float %a
  ret float %cond
}

define double @fp-armv8_vminnm_oge_rev(double %a, double %b) {
; CHECK-FAST-LABEL: "fp-armv8_vminnm_oge_rev":
; CHECK-FAST-NOT: vcmp
; CHECK-FAST: vminnm.f64
; CHECK-LABEL: "fp-armv8_vminnm_oge_rev":
; CHECK-NOT: vminnm.f64
  %cmp = fcmp oge double %a, %b
  %cond = select i1 %cmp, double %b, double %a
  ret double %cond
}

define float @fp-armv8_vminnm_u(float %a, float %b) {
; CHECK-FAST-LABEL: "fp-armv8_vminnm_u":
; CHECK-FAST-NOT: vcmp
; CHECK-FAST: vminnm.f32
; CHECK-LABEL: "fp-armv8_vminnm_u":
; CHECK-NOT: vminnm.f32
  %cmp = fcmp ult float %a, %b
  %cond = select i1 %cmp, float %a, float %b
  ret float %cond
}

define float @fp-armv8_vminnm_ule(float %a, float %b) {
; CHECK-FAST-LABEL: "fp-armv8_vminnm_ule":
; CHECK-FAST-NOT: vcmp
; CHECK-FAST: vminnm.f32
; CHECK-LABEL: "fp-armv8_vminnm_ule":
; CHECK-NOT: vminnm.f32
  %cmp = fcmp ule float %a, %b
  %cond = select i1 %cmp, float %a, float %b
  ret float %cond
}

define float @fp-armv8_vminnm_u_rev(float %a, float %b) {
; CHECK-FAST-LABEL: "fp-armv8_vminnm_u_rev":
; CHECK-FAST-NOT: vcmp
; CHECK-FAST: vminnm.f32
; CHECK-LABEL: "fp-armv8_vminnm_u_rev":
; CHECK-NOT: vminnm.f32
  %cmp = fcmp ugt float %a, %b
  %cond = select i1 %cmp, float %b, float %a
  ret float %cond
}

define double @fp-armv8_vminnm_uge_rev(double %a, double %b) {
; CHECK-FAST-LABEL: "fp-armv8_vminnm_uge_rev":
; CHECK-FAST-NOT: vcmp
; CHECK-FAST: vminnm.f64
; CHECK-LABEL: "fp-armv8_vminnm_uge_rev":
; CHECK-NOT: vminnm.f64
  %cmp = fcmp uge double %a, %b
  %cond = select i1 %cmp, double %b, double %a
  ret double %cond
}

define float @fp-armv8_vmaxnm_o(float %a, float %b) {
; CHECK-FAST-LABEL: "fp-armv8_vmaxnm_o":
; CHECK-FAST-NOT: vcmp
; CHECK-FAST: vmaxnm.f32
; CHECK-LABEL: "fp-armv8_vmaxnm_o":
; CHECK-NOT: vmaxnm.f32
  %cmp = fcmp ogt float %a, %b
  %cond = select i1 %cmp, float %a, float %b
  ret float %cond
}

define float @fp-armv8_vmaxnm_oge(float %a, float %b) {
; CHECK-FAST-LABEL: "fp-armv8_vmaxnm_oge":
; CHECK-FAST-NOT: vcmp
; CHECK-FAST: vmaxnm.f32
; CHECK-LABEL: "fp-armv8_vmaxnm_oge":
; CHECK-NOT: vmaxnm.f32
  %cmp = fcmp oge float %a, %b
  %cond = select i1 %cmp, float %a, float %b
  ret float %cond
}

define float @fp-armv8_vmaxnm_o_rev(float %a, float %b) {
; CHECK-FAST-LABEL: "fp-armv8_vmaxnm_o_rev":
; CHECK-FAST-NOT: vcmp
; CHECK-FAST: vmaxnm.f32
; CHECK-LABEL: "fp-armv8_vmaxnm_o_rev":
; CHECK-NOT: vmaxnm.f32
  %cmp = fcmp olt float %a, %b
  %cond = select i1 %cmp, float %b, float %a
  ret float %cond
}

define float @fp-armv8_vmaxnm_ole_rev(float %a, float %b) {
; CHECK-FAST-LABEL: "fp-armv8_vmaxnm_ole_rev":
; CHECK-FAST-NOT: vcmp
; CHECK-FAST: vmaxnm.f32
; CHECK-LABEL: "fp-armv8_vmaxnm_ole_rev":
; CHECK-NOT: vmaxnm.f32
  %cmp = fcmp ole float %a, %b
  %cond = select i1 %cmp, float %b, float %a
  ret float %cond
}

define float @fp-armv8_vmaxnm_u(float %a, float %b) {
; CHECK-FAST-LABEL: "fp-armv8_vmaxnm_u":
; CHECK-FAST-NOT: vcmp
; CHECK-FAST: vmaxnm.f32
; CHECK-LABEL: "fp-armv8_vmaxnm_u":
; CHECK-NOT: vmaxnm.f32
  %cmp = fcmp ugt float %a, %b
  %cond = select i1 %cmp, float %a, float %b
  ret float %cond
}

define float @fp-armv8_vmaxnm_uge(float %a, float %b) {
; CHECK-FAST-LABEL: "fp-armv8_vmaxnm_uge":
; CHECK-FAST-NOT: vcmp
; CHECK-FAST: vmaxnm.f32
; CHECK-LABEL: "fp-armv8_vmaxnm_uge":
; CHECK-NOT: vmaxnm.f32
  %cmp = fcmp uge float %a, %b
  %cond = select i1 %cmp, float %a, float %b
  ret float %cond
}

define float @fp-armv8_vmaxnm_u_rev(float %a, float %b) {
; CHECK-FAST-LABEL: "fp-armv8_vmaxnm_u_rev":
; CHECK-FAST-NOT: vcmp
; CHECK-FAST: vmaxnm.f32
; CHECK-LABEL: "fp-armv8_vmaxnm_u_rev":
; CHECK-NOT: vmaxnm.f32
  %cmp = fcmp ult float %a, %b
  %cond = select i1 %cmp, float %b, float %a
  ret float %cond
}

define double @fp-armv8_vmaxnm_ule_rev(double %a, double %b) {
; CHECK-FAST-LABEL: "fp-armv8_vmaxnm_ule_rev":
; CHECK-FAST-NOT: vcmp
; CHECK-FAST: vmaxnm.f64
; CHECK-LABEL: "fp-armv8_vmaxnm_ule_rev":
; CHECK-NOT: vmaxnm.f64
  %cmp = fcmp ule double %a, %b
  %cond = select i1 %cmp, double %b, double %a
  ret double %cond
}


declare <4 x float> @llvm.arm.neon.vminnm.v4f32(<4 x float>, <4 x float>) nounwind readnone
declare <2 x float> @llvm.arm.neon.vminnm.v2f32(<2 x float>, <2 x float>) nounwind readnone
declare <4 x float> @llvm.arm.neon.vmaxnm.v4f32(<4 x float>, <4 x float>) nounwind readnone
declare <2 x float> @llvm.arm.neon.vmaxnm.v2f32(<2 x float>, <2 x float>) nounwind readnone
