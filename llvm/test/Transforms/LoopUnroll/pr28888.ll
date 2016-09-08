; RUN: opt -loop-unroll -verify-loop-info -unroll-runtime-epilog=false -unroll-count=4 -S < %s | FileCheck %s -check-prefix=PROLOG
; RUN: opt -loop-unroll -verify-loop-info -unroll-runtime-epilog=true  -unroll-count=4 -S < %s | FileCheck %s -check-prefix=EPILOG

; PR28888
; Check that loop info is correct if we unroll an outer loop, and thus the
; remainder loop has a child loop.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; PROLOG-LABEL: @foo
; EPILOG-LABEL: @foo
define void @foo(i1 %x) #0 {
bb:
  br label %bb1

bb1:
  br label %bb2

; PROLOG: bb2.prol:
; EPILOG: bb2.epil:
bb2:
  %tmp = phi i64 [ 0, %bb1 ], [ %tmp2, %bb5 ]
  br label %bb3

bb3:
  br label %bb4

bb4:
  br i1 %x, label %bb3, label %bb5

; PROLOG: bb5.3:
; EPILOG: bb5.3:
bb5:
  %tmp2 = add nuw nsw i64 %tmp, 1
  %tmp3 = trunc i64 %tmp2 to i32
  %tmp4 = icmp eq i32 %tmp3, undef
  br i1 %tmp4, label %bb6, label %bb2

bb6:
  br label %bb1
}

attributes #0 = { "target-cpu"="x86-64" }
