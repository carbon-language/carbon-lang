; REQUIRES: asserts
; RUN: llc < %s -mtriple=thumbv8m.main-none-eabi -debug-only=machine-scheduler,post-RA-sched -print-before=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s

; CHECK-LABEL: test_misched
; Pre and post ra machine scheduling
; CHECK:  ********** MI Scheduling **********
; CHECK:  t2LDRi12
; CHECK:  Latency            : 2
; CHECK:  ********** MI Scheduling **********
; CHECK:  t2LDRi12
; CHECK:  Latency            : 2

define i32 @test_misched(i32* %ptr) "target-cpu"="cortex-m33" {
entry:
  %l = load i32, i32* %ptr
  store i32 0, i32* %ptr
  ret i32 %l
}

; CHECK-LABEL: test_rasched
; CHECK: Subtarget disables post-MI-sched.
; CHECK: ********** List Scheduling **********

define i32 @test_rasched(i32* %ptr) {
entry:
  %l = load i32, i32* %ptr
  store i32 0, i32* %ptr
  ret i32 %l
}

