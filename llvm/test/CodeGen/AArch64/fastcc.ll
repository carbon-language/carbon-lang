; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -tailcallopt | FileCheck %s -check-prefix CHECK-TAIL
; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s

; Without tailcallopt fastcc still means the caller cleans up the
; stack, so try to make sure this is respected.

define fastcc void @func_stack0() {
; CHECK-LABEL: func_stack0:
; CHECK: mov x29, sp
; CHECK-NEXT: sub sp, sp, #32

; CHECK-TAIL-LABEL: func_stack0:
; CHECK-TAIL: stp x29, x30, [sp, #-16]!
; CHECK-TAIL-NEXT: mov x29, sp
; CHECK-TAIL-NEXT: sub sp, sp, #32


  call fastcc void @func_stack8([8 x i32] undef, i32 42)
; CHECK:  bl func_stack8
; CHECK-NOT: sub sp, sp,

; CHECK-TAIL: bl func_stack8
; CHECK-TAIL: sub sp, sp, #16


  call fastcc void @func_stack32([8 x i32] undef, i128 0, i128 9)
; CHECK: bl func_stack32
; CHECK-NOT: sub sp, sp,


; CHECK-TAIL: bl func_stack32
; CHECK-TAIL: sub sp, sp, #32


  call fastcc void @func_stack0()
; CHECK: bl func_stack0
; CHECK-NOT: sub sp, sp


; CHECK-TAIL: bl func_stack0
; CHECK-TAIL-NOT: sub sp, sp

  ret void
; CHECK: mov sp, x29
; CHECK-NEXT: ldp     x29, x30, [sp], #16
; CHECK-NEXT: ret


; CHECK-TAIL: mov sp, x29
; CHECK-TAIL-NEXT: ldp     x29, x30, [sp], #16
; CHECK-TAIL-NEXT: ret
}

define fastcc void @func_stack8([8 x i32], i32 %stacked) {
; CHECK-LABEL: func_stack8:
; CHECK: stp x29, x30, [sp, #-16]!
; CHECK: mov x29, sp
; CHECK: sub sp, sp, #32


; CHECK-TAIL-LABEL: func_stack8:
; CHECK-TAIL: stp x29, x30, [sp, #-16]!
; CHECK-TAIL: mov x29, sp
; CHECK-TAIL: sub sp, sp, #32


  call fastcc void @func_stack8([8 x i32] undef, i32 42)
; CHECK:  bl func_stack8
; CHECK-NOT: sub sp, sp,


; CHECK-TAIL: bl func_stack8
; CHECK-TAIL: sub sp, sp, #16


  call fastcc void @func_stack32([8 x i32] undef, i128 0, i128 9)
; CHECK: bl func_stack32
; CHECK-NOT: sub sp, sp,


; CHECK-TAIL: bl func_stack32
; CHECK-TAIL: sub sp, sp, #32


  call fastcc void @func_stack0()
; CHECK: bl func_stack0
; CHECK-NOT: sub sp, sp

; CHECK-TAIL: bl func_stack0
; CHECK-TAIL-NOT: sub sp, sp

  ret void
; CHECK: mov sp, x29
; CHECK-NEXT: ldp     x29, x30, [sp], #16
; CHECK-NEXT: ret


; CHECK-TAIL: mov sp, x29
; CHECK-TAIL-NEXT: ldp     x29, x30, [sp], #16
; CHECK-TAIL-NEXT: ret
}

define fastcc void @func_stack32([8 x i32], i128 %stacked0, i128 %stacked1) {
; CHECK-LABEL: func_stack32:
; CHECK: mov x29, sp

; CHECK-TAIL-LABEL: func_stack32:
; CHECK-TAIL: mov x29, sp


  call fastcc void @func_stack8([8 x i32] undef, i32 42)
; CHECK:  bl func_stack8
; CHECK-NOT: sub sp, sp,

; CHECK-TAIL: bl func_stack8
; CHECK-TAIL: sub sp, sp, #16


  call fastcc void @func_stack32([8 x i32] undef, i128 0, i128 9)
; CHECK: bl func_stack32
; CHECK-NOT: sub sp, sp,


; CHECK-TAIL: bl func_stack32
; CHECK-TAIL: sub sp, sp, #32


  call fastcc void @func_stack0()
; CHECK: bl func_stack0
; CHECK-NOT: sub sp, sp


; CHECK-TAIL: bl func_stack0
; CHECK-TAIL-NOT: sub sp, sp

  ret void
; CHECK: mov sp, x29
; CHECK-NEXT: ldp     x29, x30, [sp], #16
; CHECK-NEXT: ret

; CHECK-TAIL: mov sp, x29
; CHECK-TAIL-NEXT: ldp     x29, x30, [sp], #16
; CHECK-TAIL-NEXT: ret
}
