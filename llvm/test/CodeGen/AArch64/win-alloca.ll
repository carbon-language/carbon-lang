; RUN: llc -mtriple aarch64-windows -verify-machineinstrs -filetype asm -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-OPT
; RUN: llc -mtriple aarch64-windows -verify-machineinstrs -filetype asm -o - %s -O0 | FileCheck %s

define void @func(i64 %a) {
entry:
  %0 = alloca i8, i64 %a, align 16
  call void @func2(i8* nonnull %0)
  ret void
}

declare void @func2(i8*)

; The -O0 version here ends up much less elegant, so just check the
; details of the optimized form, but check that -O0 at least emits the
; call to __chkstk.

; CHECK: add [[REG1:x[0-9]+]], x0, #15
; CHECK-OPT: lsr x15, [[REG1]], #4
; CHECK: bl __chkstk
; CHECK: mov [[REG2:x[0-9]+]], sp
; CHECK-OPT: sub [[REG3:x[0-9]+]], [[REG2]], x15, lsl #4
; CHECK-OPT: mov sp, [[REG3]]
; CHECK: bl func2
