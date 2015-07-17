; This test checks that non-instrumented allocas stay in the first basic block.
; Only first-basic-block allocas are considered stack slots, and moving them
; breaks debug info.

; RUN: opt < %s -asan -asan-module -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

define i32 @foo() sanitize_address {
entry:
  ; Regular alloca, will get instrumented (forced by the ptrtoint below).
  %instrumented = alloca i32, align 4

  ; Won't be instrumented because of asan-skip-promotable-allocas.
  %non_instrumented = alloca i32, align 4
  store i32 0, i32* %non_instrumented, align 4
  %value = load i32, i32* %non_instrumented, align 4

  %ptr = ptrtoint i32* %instrumented to i64
  ret i32 %value
}

; CHECK: entry:
; CHECK: %non_instrumented = alloca i32, align 4
; CHECK: load i32, i32* @__asan_option_detect_stack_use_after_return
