; Test that undecidable dynamic allocas are skipped by ASan.

; RUN: opt < %s -asan -asan-module -asan-instrument-allocas=1 -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define void @g(i64 %n) sanitize_address {
entry:
  %cmp = icmp sgt i64 %n, 100
  br i1 %cmp, label %do_alloca, label %done

do_alloca:
; CHECK-NOT: store i32 -892679478
  %0 = alloca i8, i64 %n, align 1
  call void @f(i8* %0)
  br label %done

done:
  ret void
}

declare void @f(i8*)

