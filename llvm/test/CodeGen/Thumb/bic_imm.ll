; RUN: llc < %s -mtriple=thumbv7-linux-gnueabi -mcpu=cortex-m0 -verify-machineinstrs | FileCheck --check-prefix CHECK-T1 %s
; RUN: llc < %s -mtriple=thumbv7-linux-gnueabi -mcpu=cortex-m3 -verify-machineinstrs | FileCheck --check-prefix CHECK-T2 %s

; CHECK-T1-LABEL: @i
; CHECK-T2-LABEL: @i
; CHECK-T1: bics r0, #275
; CHECK-T2: bic r0, r0, #275
define i32 @i(i32 %a) {
entry:
  %and = and i32 %a, -276
  ret i32 %and
}
