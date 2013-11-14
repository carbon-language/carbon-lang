; RUN: opt < %s -asan -asan-coverage=1 -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
define i32 @foo(i32* %a) sanitize_address {
entry:
  ret i32 0
}
; CHECK: define i32 @foo(i32* %a) #0 {
; CHECK: %0 = load atomic i8* @__asan_gen_cov_foo monotonic, align 1
; CHECK: %1 = icmp eq i8 0, %0
; CHECK: br i1 %1, label %2, label %3
; CHECK: call void @__sanitizer_cov(i64 ptrtoint (i32 (i32*)* @foo to i64))
; CHECK: store atomic i8 1, i8* @__asan_gen_cov_foo monotonic, align 1
