; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -tailcallopt | FileCheck %s -check-prefix CHECK-TAIL
; RUN: llc -verify-machineinstrs < %s -mtriple=arm64-none-linux-gnu -tailcallopt | FileCheck %s -check-prefix CHECK-ARM64-TAIL
; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=arm64-none-linux-gnu | FileCheck --check-prefix=CHECK-ARM64 %s

; Without tailcallopt fastcc still means the caller cleans up the
; stack, so try to make sure this is respected.

define fastcc void @func_stack0() {
; CHECK-LABEL: func_stack0:
; CHECK: sub sp, sp, #48

; CHECK-ARM64-LABEL: func_stack0:
; CHECK-ARM64: stp x29, x30, [sp, #-16]!
; CHECK-ARM64-NEXT: mov x29, sp
; CHECK-ARM64-NEXT: sub sp, sp, #32

; CHECK-TAIL-LABEL: func_stack0:
; CHECK-TAIL: sub sp, sp, #48

; CHECK-ARM64-TAIL-LABEL: func_stack0:
; CHECK-ARM64-TAIL: stp x29, x30, [sp, #-16]!
; CHECK-ARM64-TAIL-NEXT: mov x29, sp
; CHECK-ARM64-TAIL-NEXT: sub sp, sp, #32


  call fastcc void @func_stack8([8 x i32] undef, i32 42)
; CHECK:  bl func_stack8
; CHECK-NOT: sub sp, sp,

; CHECK-ARM64:  bl func_stack8
; CHECK-ARM64-NOT: sub sp, sp,

; CHECK-TAIL: bl func_stack8
; CHECK-TAIL: sub sp, sp, #16

; CHECK-ARM64-TAIL: bl func_stack8
; CHECK-ARM64-TAIL: sub sp, sp, #16


  call fastcc void @func_stack32([8 x i32] undef, i128 0, i128 9)
; CHECK: bl func_stack32
; CHECK-NOT: sub sp, sp,

; CHECK-ARM64: bl func_stack32
; CHECK-ARM64-NOT: sub sp, sp,

; CHECK-TAIL: bl func_stack32
; CHECK-TAIL: sub sp, sp, #32

; CHECK-ARM64-TAIL: bl func_stack32
; CHECK-ARM64-TAIL: sub sp, sp, #32


  call fastcc void @func_stack0()
; CHECK: bl func_stack0
; CHECK-NOT: sub sp, sp

; CHECK-ARM64: bl func_stack0
; CHECK-ARM64-NOT: sub sp, sp

; CHECK-TAIL: bl func_stack0
; CHECK-TAIL-NOT: sub sp, sp

; CHECK-ARM64-TAIL: bl func_stack0
; CHECK-ARM64-TAIL-NOT: sub sp, sp

  ret void
; CHECK: add sp, sp, #48
; CHECK-NEXT: ret

; CHECK-ARM64: mov sp, x29
; CHECK-ARM64-NEXT: ldp     x29, x30, [sp], #16
; CHECK-ARM64-NEXT: ret

; CHECK-TAIL: add sp, sp, #48
; CHECK-TAIL-NEXT: ret

; CHECK-ARM64-TAIL: mov sp, x29
; CHECK-ARM64-TAIL-NEXT: ldp     x29, x30, [sp], #16
; CHECK-ARM64-TAIL-NEXT: ret
}

define fastcc void @func_stack8([8 x i32], i32 %stacked) {
; CHECK-LABEL: func_stack8:
; CHECK: sub sp, sp, #48

; CHECK-ARM64-LABEL: func_stack8:
; CHECK-ARM64: stp x29, x30, [sp, #-16]!
; CHECK-ARM64: mov x29, sp
; CHECK-ARM64: sub sp, sp, #32

; CHECK-TAIL-LABEL: func_stack8:
; CHECK-TAIL: sub sp, sp, #48

; CHECK-ARM64-TAIL-LABEL: func_stack8:
; CHECK-ARM64-TAIL: stp x29, x30, [sp, #-16]!
; CHECK-ARM64-TAIL: mov x29, sp
; CHECK-ARM64-TAIL: sub sp, sp, #32


  call fastcc void @func_stack8([8 x i32] undef, i32 42)
; CHECK:  bl func_stack8
; CHECK-NOT: sub sp, sp,

; CHECK-ARM64:  bl func_stack8
; CHECK-ARM64-NOT: sub sp, sp,

; CHECK-TAIL: bl func_stack8
; CHECK-TAIL: sub sp, sp, #16

; CHECK-ARM64-TAIL: bl func_stack8
; CHECK-ARM64-TAIL: sub sp, sp, #16


  call fastcc void @func_stack32([8 x i32] undef, i128 0, i128 9)
; CHECK: bl func_stack32
; CHECK-NOT: sub sp, sp,

; CHECK-ARM64: bl func_stack32
; CHECK-ARM64-NOT: sub sp, sp,

; CHECK-TAIL: bl func_stack32
; CHECK-TAIL: sub sp, sp, #32

; CHECK-ARM64-TAIL: bl func_stack32
; CHECK-ARM64-TAIL: sub sp, sp, #32


  call fastcc void @func_stack0()
; CHECK: bl func_stack0
; CHECK-NOT: sub sp, sp

; CHECK-ARM64: bl func_stack0
; CHECK-ARM64-NOT: sub sp, sp

; CHECK-TAIL: bl func_stack0
; CHECK-TAIL-NOT: sub sp, sp

; CHECK-ARM64-TAIL: bl func_stack0
; CHECK-ARM64-TAIL-NOT: sub sp, sp

  ret void
; CHECK: add sp, sp, #48
; CHECK-NEXT: ret

; CHECK-ARM64: mov sp, x29
; CHECK-ARM64-NEXT: ldp     x29, x30, [sp], #16
; CHECK-ARM64-NEXT: ret

; CHECK-TAIL: add sp, sp, #64
; CHECK-TAIL-NEXT: ret

; CHECK-ARM64-TAIL: mov sp, x29
; CHECK-ARM64-TAIL-NEXT: ldp     x29, x30, [sp], #16
; CHECK-ARM64-TAIL-NEXT: ret
}

define fastcc void @func_stack32([8 x i32], i128 %stacked0, i128 %stacked1) {
; CHECK-LABEL: func_stack32:
; CHECK: sub sp, sp, #48

; CHECK-ARM64-LABEL: func_stack32:
; CHECK-ARM64: mov x29, sp

; CHECK-TAIL-LABEL: func_stack32:
; CHECK-TAIL: sub sp, sp, #48

; CHECK-ARM64-TAIL-LABEL: func_stack32:
; CHECK-ARM64-TAIL: mov x29, sp


  call fastcc void @func_stack8([8 x i32] undef, i32 42)
; CHECK:  bl func_stack8
; CHECK-NOT: sub sp, sp,

; CHECK-ARM64:  bl func_stack8
; CHECK-ARM64-NOT: sub sp, sp,

; CHECK-TAIL: bl func_stack8
; CHECK-TAIL: sub sp, sp, #16

; CHECK-ARM64-TAIL: bl func_stack8
; CHECK-ARM64-TAIL: sub sp, sp, #16


  call fastcc void @func_stack32([8 x i32] undef, i128 0, i128 9)
; CHECK: bl func_stack32
; CHECK-NOT: sub sp, sp,

; CHECK-ARM64: bl func_stack32
; CHECK-ARM64-NOT: sub sp, sp,

; CHECK-TAIL: bl func_stack32
; CHECK-TAIL: sub sp, sp, #32

; CHECK-ARM64-TAIL: bl func_stack32
; CHECK-ARM64-TAIL: sub sp, sp, #32


  call fastcc void @func_stack0()
; CHECK: bl func_stack0
; CHECK-NOT: sub sp, sp

; CHECK-ARM64: bl func_stack0
; CHECK-ARM64-NOT: sub sp, sp

; CHECK-TAIL: bl func_stack0
; CHECK-TAIL-NOT: sub sp, sp

; CHECK-ARM64-TAIL: bl func_stack0
; CHECK-ARM64-TAIL-NOT: sub sp, sp

  ret void
; CHECK: add sp, sp, #48
; CHECK-NEXT: ret

; CHECK-ARM64: mov sp, x29
; CHECK-ARM64-NEXT: ldp     x29, x30, [sp], #16
; CHECK-ARM64-NEXT: ret

; CHECK-TAIL: add sp, sp, #80
; CHECK-TAIL-NEXT: ret

; CHECK-ARM64-TAIL: mov sp, x29
; CHECK-ARM64-TAIL-NEXT: ldp     x29, x30, [sp], #16
; CHECK-ARM64-TAIL-NEXT: ret
}
