; RUN: llc -mcpu=swift -mtriple=thumbv7s-apple-ios -o - %s | FileCheck %s --check-prefix=CHECK-STRIDE4
; RUN: llc -mcpu=cortex-a57 -mtriple=thumbv7-linux-gnueabihf -o - %s | FileCheck %s --check-prefix=CHECK-GENERIC

define void @test_reg_stride(double %a, double %b) {
; CHECK-STRIDE4-LABEL: test_reg_stride:
; CHECK-STRIDE4-DAG: vmov d16, r
; CHECK-STRIDE4-DAG: vmov d18, r

; CHECK-GENERIC-LABEL: test_reg_stride:
; CHECK-GENERIC-DAG: vmov.f64 d16, {{d[01]}}
; CHECK-GENERIC-DAG: vmov.f64 d17, {{d[01]}}

  call void asm "", "~{r0},~{r1},~{d0},~{d1}"()
  call arm_aapcs_vfpcc void @eat_doubles(double %a, double %b)
  ret void
}

define void @test_stride_minsize(float %a, float %b) minsize {
; CHECK-STRIDE4-LABEL: test_stride_minsize:
; CHECK-STRIDE4: vmov d2, {{r[01]}}
; CHECK-STRIDE4: vmov d3, {{r[01]}}

; CHECK-GENERIC-LABEL: test_stride_minsize:
; CHECK-GENERIC-DAG: vmov.f32 s4, {{s[01]}}
; CHECK-GENERIC-DAG: vmov.f32 s6, {{s[01]}}
  call void asm "", "~{r0},~{r1},~{s0},~{s1},~{d0},~{d1}"()
  call arm_aapcs_vfpcc void @eat_floats(float %a, float %b)
  ret void
}


declare arm_aapcs_vfpcc void @eat_doubles(double, double)
declare arm_aapcs_vfpcc void @eat_floats(float, float)
