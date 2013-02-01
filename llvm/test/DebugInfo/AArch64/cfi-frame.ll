; RUN: llc -verify-machineinstrs -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -disable-fp-elim < %s | FileCheck %s --check-prefix=CHECK-WITH-FP

@bigspace = global [8 x i64] zeroinitializer

declare void @use_addr(i8*)

define void @test_frame([8 x i64] %val) {
; CHECK: test_frame:
; CHECK: .cfi_startproc

  %var = alloca i8, i32 1000000
; CHECK: sub sp, sp, #[[SP_INIT_ADJ:[0-9]+]]
; CHECK-NEXT: .Ltmp
; CHECK-NEXT: .cfi_def_cfa sp, [[SP_INIT_ADJ]]

; Make sure the prologue is reasonably efficient
; CHECK-NEXT: stp x29, x30, [sp,
; CHECK-NEXT: stp x25, x26, [sp,
; CHECK-NEXT: stp x23, x24, [sp,
; CHECK-NEXT: stp x21, x22, [sp,
; CHECK-NEXT: stp x19, x20, [sp,
; CHECK-NEXT: sub sp, sp, #160
; CHECK-NEXT: sub sp, sp, #244, lsl #12
; CHECK-NEXT: .Ltmp
; CHECK-NEXT: .cfi_def_cfa sp, 1000080
; CHECK-NEXT: .Ltmp
; CHECK-NEXT: .cfi_offset x30, -8
; CHECK-NEXT: .Ltmp
; CHECK-NEXT: .cfi_offset x29, -16
; [...]
; CHECK: .cfi_offset x19, -80

; CHECK: bl use_addr
  call void @use_addr(i8* %var)

  store [8 x i64] %val, [8 x i64]* @bigspace
  ret void
; CHECK: ret
; CHECK: .cfi_endproc
}

; CHECK-WITH-FP: test_frame:

; CHECK-WITH-FP: sub sp, sp, #[[SP_INIT_ADJ:[0-9]+]]
; CHECK-WITH-FP-NEXT: .Ltmp
; CHECK-WITH-FP-NEXT: .cfi_def_cfa sp, [[SP_INIT_ADJ]]

; CHECK-WITH-FP: stp x29, x30, [sp, [[OFFSET:#[0-9]+]]]
; CHECK-WITH-FP-NEXT: add x29, sp, [[OFFSET]]
; CHECK-WITH-FP-NEXT: .Ltmp
; CHECK-WITH-FP-NEXT: .cfi_def_cfa x29, 16

  ; We shouldn't emit any kind of update for the second stack adjustment if the
  ; FP is in use.
; CHECK-WITH-FP-NOT: .cfi_def_cfa_offset

; CHECK-WITH-FP: bl use_addr
