; RUN: opt < %s -scalarrepl | llvm-dis
; rdar://6808691
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "x86_64-apple-darwin9.0"
	type <{ i32, i16, i8, i8, i64, i64, i16, [0 x i16] }>		

define i32 @foo() {
entry:
	%.compoundliteral = alloca %0		
	%tmp228 = getelementptr %0* %.compoundliteral, i32 0, i32 7
	%tmp229 = bitcast [0 x i16]* %tmp228 to i8*		
	call void @llvm.memset.i64(i8* %tmp229, i8 0, i64 0, i32 2)
	unreachable
}

declare void @llvm.memset.i64(i8* nocapture, i8, i64, i32) nounwind
