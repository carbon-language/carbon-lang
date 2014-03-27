; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s

define i128 @test_i128_lsl(i128 %a, i32 %shift) {
; CHECK-LABEL: test_i128_lsl:

  %sh_prom = zext i32 %shift to i128
  %shl = shl i128 %a, %sh_prom

; CHECK: movz [[SIXTYFOUR:x[0-9]+]], #64
; CHECK-NEXT: sub [[REVSHAMT:x[0-9]+]], [[SIXTYFOUR]], [[SHAMT_32:w[0-9]+]], uxtw
; CHECK-NEXT: lsr [[TMP1:x[0-9]+]], [[LO:x[0-9]+]], [[REVSHAMT]]
; CHECK: lsl [[TMP2:x[0-9]+]], [[HI:x[0-9]+]], [[SHAMT:x[0-9]+]]
; CHECK-NEXT: orr [[FALSEVAL:x[0-9]+]], [[TMP1]], [[TMP2]]
; CHECK-NEXT: sub [[EXTRASHAMT:x[0-9]+]], [[SHAMT]], #64
; CHECK-NEXT: lsl [[TMP3:x[0-9]+]], [[LO]], [[EXTRASHAMT]]
; CHECK-NEXT: cmp [[EXTRASHAMT]], #0
; CHECK-NEXT: csel [[RESULTHI:x[0-9]+]], [[TMP3]], [[FALSEVAL]], ge
; CHECK-NEXT: lsl [[TMP4:x[0-9]+]], [[LO]], [[SHAMT]]
; CHECK-NEXT: csel [[RESULTLO:x[0-9]+]], xzr, [[TMP4]], ge

  ret i128 %shl
}

define i128 @test_i128_shr(i128 %a, i32 %shift) {
; CHECK-LABEL: test_i128_shr:

  %sh_prom = zext i32 %shift to i128
  %shr = lshr i128 %a, %sh_prom

; CHECK: movz [[SIXTYFOUR]], #64
; CHECK-NEXT: sub [[REVSHAMT:x[0-9]+]], [[SIXTYFOUR]], [[SHAMT_32:w[0-9]+]], uxtw
; CHECK-NEXT: lsl [[TMP2:x[0-9]+]], [[HI:x[0-9]+]], [[REVSHAMT]]
; CHECK: lsr [[TMP1:x[0-9]+]], [[LO:x[0-9]+]], [[SHAMT:x[0-9]+]]
; CHECK-NEXT: orr [[FALSEVAL:x[0-9]+]], [[TMP1]], [[TMP2]]
; CHECK-NEXT: sub [[EXTRASHAMT:x[0-9]+]], [[SHAMT]], #64
; CHECK-NEXT: lsr [[TRUEVAL:x[0-9]+]], [[HI]], [[EXTRASHAMT]]
; CHECK-NEXT: cmp [[EXTRASHAMT]], #0
; CHECK-NEXT: csel [[RESULTLO:x[0-9]+]], [[TRUEVAL]], [[FALSEVAL]], ge
; CHECK-NEXT: lsr [[TMP3:x[0-9]+]], [[HI]], [[SHAMT]]
; CHECK-NEXT: csel [[RESULTHI:x[0-9]+]], xzr, [[TMP3]], ge

  ret i128 %shr
}
