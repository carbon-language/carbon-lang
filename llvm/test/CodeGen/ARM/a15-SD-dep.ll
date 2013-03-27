; RUN: llc -O1 -mcpu=cortex-a15 -mtriple=armv7-linux-gnueabi -disable-a15-sd-optimization -verify-machineinstrs < %s  | FileCheck -check-prefix=DISABLED %s
; RUN: llc -O1 -mcpu=cortex-a15 -mtriple=armv7-linux-gnueabi -verify-machineinstrs < %s | FileCheck -check-prefix=ENABLED %s

; CHECK-ENABLED: t1:
; CHECK-DISABLED: t1:
define <2 x float> @t1(float %f) {
  ; CHECK-ENABLED: vdup.32 d{{[0-9]*}}, d0[0]
  ; CHECK-DISABLED-NOT: vdup.32 d{{[0-9]*}}, d0[0]
  %i1 = insertelement <2 x float> undef, float %f, i32 1
  %i2 = fadd <2 x float> %i1, %i1
  ret <2 x float> %i2
}

; CHECK-ENABLED: t2:
; CHECK-DISABLED: t2:
define <4 x float> @t2(float %g, float %f) {
  ; CHECK-ENABLED: vdup.32 q{{[0-9]*}}, d0[0]
  ; CHECK-DISABLED-NOT: vdup.32 d{{[0-9]*}}, d0[0]
  %i1 = insertelement <4 x float> undef, float %f, i32 1
  %i2 = fadd <4 x float> %i1, %i1
  ret <4 x float> %i2
}

; CHECK-ENABLED: t3:
; CHECK-DISABLED: t3:
define arm_aapcs_vfpcc <2 x float> @t3(float %f) {
  ; CHECK-ENABLED: vdup.32 d{{[0-9]*}}, d0[0] 
  ; CHECK-DISABLED-NOT: vdup.32 d{{[0-9]*}}, d0[0]
  %i1 = insertelement <2 x float> undef, float %f, i32 1
  %i2 = fadd <2 x float> %i1, %i1
  ret <2 x float> %i2
}

; CHECK-ENABLED: t4:
; CHECK-DISABLED: t4:
define <2 x float> @t4(float %f) {
  ; CHECK-ENABLED: vdup.32 d{{[0-9]*}}, d0[0]
  ; CHECK-DISABLED-NOT: vdup
  %i1 = insertelement <2 x float> undef, float %f, i32 1
  br label %b

  ; Block %b has an S-reg as live-in.
b:
  %i2 = fadd <2 x float> %i1, %i1
  ret <2 x float> %i2
}

; CHECK-ENABLED: t5:
; CHECK-DISABLED: t5:
define arm_aapcs_vfpcc <4 x float> @t5(<4 x float> %q, float %f) {
  ; CHECK-ENABLED: vdup.32 d{{[0-9]*}}, d{{[0-9]*}}[0]
  ; CHECK-ENABLED: vadd.f32
  ; CHECK-ENABLED-NEXT: bx lr
  ; CHECK-DISABLED-NOT: vdup
  %i1 = insertelement <4 x float> %q, float %f, i32 1
  %i2 = fadd <4 x float> %i1, %i1
  ret <4 x float> %i2
}
