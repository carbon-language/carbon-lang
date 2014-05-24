; RUN: llc -verify-machineinstrs < %s -mtriple=arm64-none-linux-gnu -tailcallopt | FileCheck %s --check-prefix=CHECK-ARM64

; This test is designed to be run in the situation where the
; call-frame is not reserved (hence disable-fp-elim), but where
; callee-pop can occur (hence tailcallopt).

declare fastcc void @will_pop([8 x i32], i32 %val)

define fastcc void @foo(i32 %in) {
; CHECK-LABEL: foo:

  %addr = alloca i8, i32 %in

; Normal frame setup stuff:
; CHECK: sub sp, sp,
; CHECK: stp x29, x30
; CHECK-ARM64: stp     x29, x30, [sp, #-16]!
; CHECK-ARM64: mov     x29, sp

; Reserve space for call-frame:
; CHECK: sub sp, sp, #16
; CHECK-ARM64: sub sp, sp, #16

  call fastcc void @will_pop([8 x i32] undef, i32 42)
; CHECK: bl will_pop
; CHECK-ARM64: bl will_pop

; Since @will_pop is fastcc with tailcallopt, it will put the stack
; back where it needs to be, we shouldn't duplicate that
; CHECK-NOT: sub sp, sp, #16
; CHECK-NOT: add sp, sp,
; CHECK-ARM64-NOT: sub sp, sp, #16
; CHECK-ARM64-NOT: add sp, sp,

; CHECK: ldp x29, x30
; CHECK: add sp, sp,
; CHECK-ARM64: mov     sp, x29
; CHECK-ARM64: ldp     x29, x30, [sp], #16
  ret void
}

declare void @wont_pop([8 x i32], i32 %val)

define void @foo1(i32 %in) {
; CHECK-LABEL: foo1:

  %addr = alloca i8, i32 %in
; Normal frame setup again
; CHECK: sub sp, sp,
; CHECK: stp x29, x30
; CHECK-ARM64: stp     x29, x30, [sp, #-16]!
; CHECK-ARM64: mov     x29, sp

; Reserve space for call-frame
; CHECK: sub sp, sp, #16
; CHECK-ARM64: sub sp, sp, #16

  call void @wont_pop([8 x i32] undef, i32 42)
; CHECK: bl wont_pop
; CHECK-ARM64: bl wont_pop

; This time we *do* need to unreserve the call-frame
; CHECK: add sp, sp, #16
; CHECK-ARM64: add sp, sp, #16

; Check for epilogue (primarily to make sure sp spotted above wasn't
; part of it).
; CHECK: ldp x29, x30
; CHECK: add sp, sp,
; CHECK-ARM64: mov     sp, x29
; CHECK-ARM64: ldp     x29, x30, [sp], #16
  ret void
}
