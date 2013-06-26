; Test the -asan-keep-uninstrumented-functions flag: FOO should get cloned
; RUN: opt < %s -asan -asan-keep-uninstrumented-functions -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = global i32 0, align 4

define i32 @main() sanitize_address {
entry:
  tail call void @FOO(i32* @a)
  ret i32 0
}

define void @FOO(i32* nocapture %x) sanitize_address {
entry:
  store i32 1, i32* %x, align 4
  ret void
}

; main should not be cloned since it is not being instrumented by asan.
; CHECK-NOT: NOASAN_main
; CHECK: define void @FOO{{.*}} section "ASAN"
; CHECK: define void @NOASAN_FOO{{.*}} section "NOASAN"
