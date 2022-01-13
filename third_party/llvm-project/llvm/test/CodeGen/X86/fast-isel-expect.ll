; RUN: llc < %s -O0 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "i686-unknown-linux-gnu"

@glbl = extern_weak constant i8

declare i64 @llvm.expect.i64(i64, i64)

define void @test() {
; CHECK: movl $glbl
  %tmp = call i64 @llvm.expect.i64(i64 zext (i1 icmp eq (i8* @glbl, i8* null) to i64), i64 0)
  %tmp2 = icmp ne i64 %tmp, 0
  br i1 %tmp2, label %bb1, label %bb2

bb1:
  unreachable

bb2:
  unreachable
}
