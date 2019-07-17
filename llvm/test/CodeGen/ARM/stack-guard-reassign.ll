; RUN: llc -O0 --frame-pointer=none -mtriple=arm-- -o - %S/../Inputs/stack-guard-reassign.ll | FileCheck %s

; Verify that the offset assigned to the stack protector is at the top of the
; frame, covering the locals.
; CHECK-LABEL: fn:
; CHECK:      sub sp, sp, #40
; CHECK-NEXT: sub sp, sp, #65536
; CHECK-NEXT: add r1, sp, #28
; CHECK-NEXT: ldr r2, .LCPI0_0
; CHECK-NEXT: ldr r3, [r2]
; CHECK-NEXT: str r3, [r1]
; CHECK-NEXT: str r0, [sp, #32]
; CHECK: .LCPI0_0:
; CHECK-NEXT: .long __stack_chk_guard
