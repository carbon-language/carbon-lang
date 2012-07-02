; RUN: opt < %s -simplify-libcalls -S | grep "llvm.memset"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-pc-linux-gnu"

define void @a(i8* %x) nounwind {
entry:
	%call = call i8* @memset(i8* %x, i32 1, i64 100)		; <i8*> [#uses=0]
	ret void
}

declare i8* @memset(i8*, i32, i64)

