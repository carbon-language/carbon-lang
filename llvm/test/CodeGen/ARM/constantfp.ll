; RUN: llc -mtriple=armv7 -mattr=+neon -mcpu=swift %s -o - | FileCheck %s
; RUN: llc -mtriple=armv7 -mattr=+neon -mcpu=cortex-a8 %s -o - | FileCheck --check-prefix=CHECK-NONEONFP %s
; RUN: llc -mtriple=armv7 -mattr=-neon -mcpu=cortex-a8 %s -o - | FileCheck --check-prefix=CHECK-NONEON %s

define arm_aapcs_vfpcc float @test_vmov_f32() {
; CHECK-LABEL: test_vmov_f32:
; CHECK: vmov.f32 d0, #1.0

; CHECK-NONEONFP: vmov.f32 s0, #1.0
  ret float 1.0
}

define arm_aapcs_vfpcc float @test_vmov_imm() {
; CHECK-LABEL: test_vmov_imm:
; CHECK: vmov.i32 d0, #0

; CHECK-NONEON-LABEL: test_vmov_imm:
; CHECK_NONEON: vldr s0, {{.?LCPI[0-9]+_[0-9]+}}
  ret float 0.0
}

define arm_aapcs_vfpcc float @test_vmvn_imm() {
; CHECK-LABEL: test_vmvn_imm:
; CHECK: vmvn.i32 d0, #0xb0000000

; CHECK-NONEON-LABEL: test_vmvn_imm:
; CHECK_NONEON: vldr s0, {{.?LCPI[0-9]+_[0-9]+}}
  ret float 8589934080.0
}

define arm_aapcs_vfpcc double @test_vmov_f64() {
; CHECK-LABEL: test_vmov_f64:
; CHECK: vmov.f64 d0, #1.0

; CHECK-NONEON-LABEL: test_vmov_f64:
; CHECK_NONEON: vmov.f64 d0, #1.0

  ret double 1.0
}

define arm_aapcs_vfpcc double @test_vmov_double_imm() {
; CHECK-LABEL: test_vmov_double_imm:
; CHECK: vmov.i32 d0, #0

; CHECK-NONEON-LABEL: test_vmov_double_imm:
; CHECK_NONEON: vldr d0, {{.?LCPI[0-9]+_[0-9]+}}
  ret double 0.0
}

define arm_aapcs_vfpcc double @test_vmvn_double_imm() {
; CHECK-LABEL: test_vmvn_double_imm:
; CHECK: vmvn.i32 d0, #0xb0000000

; CHECK-NONEON-LABEL: test_vmvn_double_imm:
; CHECK_NONEON: vldr d0, {{.?LCPI[0-9]+_[0-9]+}}
  ret double 0x4fffffff4fffffff
}

; Make sure we don't ignore the high half of 64-bit values when deciding whether
; a vmov/vmvn is possible.
define arm_aapcs_vfpcc double @test_notvmvn_double_imm() {
; CHECK-LABEL: test_notvmvn_double_imm:
; CHECK: vldr d0, {{.?LCPI[0-9]+_[0-9]+}}

; CHECK-NONEON-LABEL: test_notvmvn_double_imm:
; CHECK_NONEON: vldr d0, {{.?LCPI[0-9]+_[0-9]+}}
  ret double 0x4fffffffffffffff
}
