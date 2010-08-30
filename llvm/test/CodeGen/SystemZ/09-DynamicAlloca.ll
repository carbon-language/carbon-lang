; RUN: llc < %s

target datalayout = "E-p:64:64:64-i8:8:16-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-a0:16:16"
target triple = "s390x-ibm-linux"

define void @foo(i64 %N) nounwind {
entry:
	%N3 = trunc i64 %N to i32		; <i32> [#uses=1]
	%vla = alloca i8, i32 %N3, align 2		; <i8*> [#uses=1]
	call void @bar(i8* %vla) nounwind
	ret void
}

declare void @bar(i8*)
