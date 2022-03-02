; This test checks that we are not instrumenting sanitizer code.
; RUN: opt < %s -passes='asan-pipeline' -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function with sanitize_address is instrumented.
; Function Attrs: nounwind uwtable
define void @instr_sa(i32* %a) sanitize_address {
entry:
  %tmp1 = load i32, i32* %a, align 4
  %tmp2 = add i32 %tmp1,  1
  store i32 %tmp2, i32* %a, align 4
  ret void
}

; CHECK-LABEL: @instr_sa
; CHECK: call void @__asan_report_load


; Function with disable_sanitizer_instrumentation is not instrumented.
; Function Attrs: nounwind uwtable
define void @noinstr_dsi(i32* %a) disable_sanitizer_instrumentation {
entry:
  %tmp1 = load i32, i32* %a, align 4
  %tmp2 = add i32 %tmp1,  1
  store i32 %tmp2, i32* %a, align 4
  ret void
}

; CHECK-LABEL: @noinstr_dsi
; CHECK-NOT: call void @__asan_report_load


; disable_sanitizer_instrumentation takes precedence over sanitize_address.
; Function Attrs: nounwind uwtable
define void @noinstr_dsi_sa(i32* %a) disable_sanitizer_instrumentation sanitize_address {
entry:
  %tmp1 = load i32, i32* %a, align 4
  %tmp2 = add i32 %tmp1,  1
  store i32 %tmp2, i32* %a, align 4
  ret void
}

; CHECK-LABEL: @noinstr_dsi_sa
; CHECK-NOT: call void @__asan_report_load

