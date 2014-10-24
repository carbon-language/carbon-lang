; RUN: opt < %s -msan -msan-check-constant-shadow=1 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Test that returning a literal undef from main() triggers an MSan warning.

define i32 @main() nounwind uwtable sanitize_memory {
entry:
  ret i32 undef
}

; CHECK-LABEL: @main
; CHECK: call void @__msan_warning_noreturn
; CHECK: ret i32 undef
