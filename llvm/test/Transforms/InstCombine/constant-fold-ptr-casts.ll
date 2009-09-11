; RUN: opt < %s -instcombine -S | grep {ret i32 2143034560}

; Instcombine should be able to completely fold this code.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"

@bar = constant [3 x i64] [i64 9220983451228067448, i64 9220983451228067449, i64 9220983450959631991], align 8

define i32 @foo() nounwind {
entry:
	%tmp87.2 = load i64* inttoptr (i32 add (i32 16, i32 ptrtoint ([3 x i64]* @bar to i32)) to i64*), align 8
	%t0 = bitcast i64 %tmp87.2 to double
	%tmp9192.2 = fptrunc double %t0 to float
	%t1 = bitcast float %tmp9192.2 to i32
	ret i32 %t1
}

