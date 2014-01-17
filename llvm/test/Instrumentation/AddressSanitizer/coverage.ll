; RUN: opt < %s -asan -asan-coverage=1 -S | FileCheck %s --check-prefix=CHECK1
; RUN: opt < %s -asan -asan-coverage=2 -S | FileCheck %s --check-prefix=CHECK2
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
define void @foo(i32* %a) sanitize_address {
entry:
  %tobool = icmp eq i32* %a, null
  br i1 %tobool, label %if.end, label %if.then

  if.then:                                          ; preds = %entry
  store i32 0, i32* %a, align 4
  br label %if.end

  if.end:                                           ; preds = %entry, %if.then
  ret void
}
; CHECK1-LABEL: define void @foo
; CHECK1: %0 = load atomic i8* @__asan_gen_cov_foo monotonic, align 1
; CHECK1: %1 = icmp eq i8 0, %0
; CHECK1: br i1 %1, label %2, label %3
; CHECK1: call void @__sanitizer_cov
; CHECK1-NOT: call void @__sanitizer_cov
; CHECK1: store atomic i8 1, i8* @__asan_gen_cov_foo monotonic, align 1

; CHECK2-LABEL: define void @foo
; CHECK2: call void @__sanitizer_cov
; CHECK2: call void @__sanitizer_cov
; CHECK2: call void @__sanitizer_cov
; CHECK2-NOT: call void @__sanitizer_cov
; CHECK2: ret void
