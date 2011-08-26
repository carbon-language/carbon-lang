; RUN: llc < %s -mtriple=armv7-apple-ios | FileCheck %s -check-prefix=ARM
; RUN: llc < %s -mtriple=thumbv7-apple-ios | FileCheck %s -check-prefix=THUMBTWO
; RUN: llc < %s -mtriple=thumbv6-apple-ios | FileCheck %s -check-prefix=THUMBONE

define void @test1(i32* %ptr, i32 %val1) {
; ARM: test1
; ARM: dmb ish
; ARM-NEXT: str
; ARM-NEXT: dmb ish
; THUMBONE: test1
; THUMBONE: __sync_lock_test_and_set_4
; THUMBTWO: test1
; THUMBTWO: dmb ish
; THUMBTWO-NEXT: str
; THUMBTWO-NEXT: dmb ish
  store atomic i32 %val1, i32* %ptr seq_cst, align 4
  ret void
}

define i32 @test2(i32* %ptr) {
; ARM: test2
; ARM: ldr
; ARM-NEXT: dmb ish
; THUMBONE: test2
; THUMBONE: __sync_val_compare_and_swap_4
; THUMBTWO: test2
; THUMBTWO: ldr
; THUMBTWO-NEXT: dmb ish
  %val = load atomic i32* %ptr seq_cst, align 4
  ret i32 %val
}
