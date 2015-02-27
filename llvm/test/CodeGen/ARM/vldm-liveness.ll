; RUN: llc -mtriple thumbv7-apple-ios -verify-machineinstrs -o - %s | FileCheck %s

; ARM load store optimizer was dealing with a sequence like:
;     s1 = VLDRS [r0, 1], Q0<imp-def>
;     s3 = VLDRS [r0, 2], Q0<imp-use,kill>, Q0<imp-def>
;     s0 = VLDRS [r0, 0], Q0<imp-use,kill>, Q0<imp-def>
;     s2 = VLDRS [r0, 4], Q0<imp-use,kill>, Q0<imp-def>
;
; It decided to combine the {s0, s1} loads into a single instruction in the
; third position. However, this leaves the instruction defining s3 with a stray
; imp-use of Q0, which is undefined.
;
; The verifier catches this, so this test just makes sure that appropriate
; liveness flags are added.
;
; I believe the change will be tested as long as the vldmia is not the first of
; the loads. Earlier optimisations may perturb the output over time, but
; fiddling the indices should be sufficient to restore the test.

define arm_aapcs_vfpcc <4 x float> @foo(float* %ptr) {
; CHECK-LABEL: foo:
; CHECK: vldr s3, [r0, #8]
; CHECK: vldmia r0, {s0, s1}
; CHECK: vldr s2, [r0, #16]
   %off0 = getelementptr float, float* %ptr, i32 0
   %val0 = load float, float* %off0
   %off1 = getelementptr float, float* %ptr, i32 1
   %val1 = load float, float* %off1
   %off4 = getelementptr float, float* %ptr, i32 4
   %val4 = load float, float* %off4
   %off2 = getelementptr float, float* %ptr, i32 2
   %val2 = load float, float* %off2

   %vec1 = insertelement <4 x float> undef, float %val0, i32 0
   %vec2 = insertelement <4 x float> %vec1, float %val1, i32 1
   %vec3 = insertelement <4 x float> %vec2, float %val4, i32 2
   %vec4 = insertelement <4 x float> %vec3, float %val2, i32 3

   ret <4 x float> %vec4
}
