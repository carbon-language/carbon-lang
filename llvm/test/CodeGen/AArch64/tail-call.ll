; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -tailcallopt | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=arm64-none-linux-gnu -tailcallopt | FileCheck --check-prefix=CHECK-ARM64 %s

declare fastcc void @callee_stack0()
declare fastcc void @callee_stack8([8 x i32], i64)
declare fastcc void @callee_stack16([8 x i32], i64, i64)

define fastcc void @caller_to0_from0() nounwind {
; CHECK-LABEL: caller_to0_from0:
; CHECK-NEXT: // BB

; CHECK-ARM64-LABEL: caller_to0_from0:
; CHECK-ARM64-NEXT: // BB

  tail call fastcc void @callee_stack0()
  ret void

; CHECK-NEXT: b callee_stack0

; CHECK-ARM64-NEXT: b callee_stack0
}

define fastcc void @caller_to0_from8([8 x i32], i64) {
; CHECK-LABEL: caller_to0_from8:

; CHECK-ARM64-LABEL: caller_to0_from8:

  tail call fastcc void @callee_stack0()
  ret void

; CHECK: add sp, sp, #16
; CHECK-NEXT: b callee_stack0

; CHECK-ARM64: add sp, sp, #16
; CHECK-ARM64-NEXT: b callee_stack0
}

define fastcc void @caller_to8_from0() {
; CHECK-LABEL: caller_to8_from0:
; CHECK: sub sp, sp, #32

; CHECK-ARM64-LABEL: caller_to8_from0:
; CHECK-ARM64: sub sp, sp, #32

; Key point is that the "42" should go #16 below incoming stack
; pointer (we didn't have arg space to reuse).
  tail call fastcc void @callee_stack8([8 x i32] undef, i64 42)
  ret void

; CHECK: str {{x[0-9]+}}, [sp, #16]
; CHECK-NEXT: add sp, sp, #16
; CHECK-NEXT: b callee_stack8

; CHECK-ARM64: str {{x[0-9]+}}, [sp, #16]!
; CHECK-ARM64-NEXT: b callee_stack8
}

define fastcc void @caller_to8_from8([8 x i32], i64 %a) {
; CHECK-LABEL: caller_to8_from8:
; CHECK: sub sp, sp, #16

; CHECK-ARM64-LABEL: caller_to8_from8:
; CHECK-ARM64: sub sp, sp, #16

; Key point is that the "%a" should go where at SP on entry.
  tail call fastcc void @callee_stack8([8 x i32] undef, i64 42)
  ret void

; CHECK: str {{x[0-9]+}}, [sp, #16]
; CHECK-NEXT: add sp, sp, #16
; CHECK-NEXT: b callee_stack8

; CHECK-ARM64: str {{x[0-9]+}}, [sp, #16]!
; CHECK-ARM64-NEXT: b callee_stack8
}

define fastcc void @caller_to16_from8([8 x i32], i64 %a) {
; CHECK-LABEL: caller_to16_from8:
; CHECK: sub sp, sp, #16

; CHECK-ARM64-LABEL: caller_to16_from8:
; CHECK-ARM64: sub sp, sp, #16

; Important point is that the call reuses the "dead" argument space
; above %a on the stack. If it tries to go below incoming-SP then the
; callee will not deallocate the space, even in fastcc.
  tail call fastcc void @callee_stack16([8 x i32] undef, i64 42, i64 2)

; CHECK: str {{x[0-9]+}}, [sp, #24]
; CHECK: str {{x[0-9]+}}, [sp, #16]
; CHECK-NEXT: add sp, sp, #16
; CHECK-NEXT: b callee_stack16

; CHECK-ARM64: stp {{x[0-9]+}}, {{x[0-9]+}}, [sp, #16]
; CHECK-ARM64-NEXT: add sp, sp, #16
; CHECK-ARM64-NEXT: b callee_stack16
  ret void
}


define fastcc void @caller_to8_from24([8 x i32], i64 %a, i64 %b, i64 %c) {
; CHECK-LABEL: caller_to8_from24:
; CHECK: sub sp, sp, #16

; CHECK-ARM64-LABEL: caller_to8_from24:
; CHECK-ARM64: sub sp, sp, #16

; Key point is that the "%a" should go where at #16 above SP on entry.
  tail call fastcc void @callee_stack8([8 x i32] undef, i64 42)
  ret void

; CHECK: str {{x[0-9]+}}, [sp, #32]
; CHECK-NEXT: add sp, sp, #32
; CHECK-NEXT: b callee_stack8

; CHECK-ARM64: str {{x[0-9]+}}, [sp, #32]!
; CHECK-ARM64-NEXT: b callee_stack8
}


define fastcc void @caller_to16_from16([8 x i32], i64 %a, i64 %b) {
; CHECK-LABEL: caller_to16_from16:
; CHECK: sub sp, sp, #16

; CHECK-ARM64-LABEL: caller_to16_from16:
; CHECK-ARM64: sub sp, sp, #16

; Here we want to make sure that both loads happen before the stores:
; otherwise either %a or %b will be wrongly clobbered.
  tail call fastcc void @callee_stack16([8 x i32] undef, i64 %b, i64 %a)
  ret void

; CHECK: ldr x0,
; CHECK: ldr x1,
; CHECK: str x1,
; CHECK: str x0,

; CHECK-NEXT: add sp, sp, #16
; CHECK-NEXT: b callee_stack16

; CHECK-ARM64: ldp {{x[0-9]+}}, {{x[0-9]+}}, [sp, #16]
; CHECK-ARM64: stp {{x[0-9]+}}, {{x[0-9]+}}, [sp, #16]
; CHECK-ARM64-NEXT: add sp, sp, #16
; CHECK-ARM64-NEXT: b callee_stack16
}
