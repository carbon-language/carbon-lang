; RUN: llc -mtriple arm-gnueabi -mattr=+hwdiv-arm -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,HWDIV
; RUN: llc -mtriple arm-gnueabi -mattr=-hwdiv-arm -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,SOFT-AEABI
; RUN: llc -mtriple arm-gnu -mattr=+hwdiv-arm -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,HWDIV
; RUN: llc -mtriple arm-gnu -mattr=-hwdiv-arm -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,SOFT-DEFAULT

define arm_aapcscc i32 @test_sdiv_i32(i32 %a, i32 %b) {
; CHECK-LABEL: test_sdiv_i32:
; HWDIV: sdiv r0, r0, r1
; SOFT-AEABI: blx __aeabi_idiv
; SOFT-DEFAULT: blx __divsi3
  %r = sdiv i32 %a, %b
  ret i32 %r
}

define arm_aapcscc i32 @test_udiv_i32(i32 %a, i32 %b) {
; CHECK-LABEL: test_udiv_i32:
; HWDIV: udiv r0, r0, r1
; SOFT-AEABI: blx __aeabi_uidiv
; SOFT-DEFAULT: blx __udivsi3
  %r = udiv i32 %a, %b
  ret i32 %r
}

