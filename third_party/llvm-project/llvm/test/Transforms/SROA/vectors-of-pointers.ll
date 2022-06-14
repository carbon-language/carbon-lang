; RUN: opt < %s -sroa

; Make sure we don't crash on this one.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

define void @foo(i1 %c1, i1 %c2) {
entry:
  %Args.i = alloca <2 x i32*>, align 16
  br i1 %c1, label %bb0.exit158, label %if.then.i.i.i.i.i138

if.then.i.i.i.i.i138:
  ret void

bb0.exit158:
  br i1 %c2, label %bb0.exit257, label %if.then.i.i.i.i.i237

if.then.i.i.i.i.i237:
  unreachable

bb0.exit257:
  %0 = load <2 x i32*>, <2 x i32*>* %Args.i, align 16
  ret void
}
