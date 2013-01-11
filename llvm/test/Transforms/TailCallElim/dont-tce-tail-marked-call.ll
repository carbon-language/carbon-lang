; RUN: opt < %s -tailcallelim -S | FileCheck %s

declare void @bar(i32*)

define i32 @foo(i32 %N) {
	%A = alloca i32, i32 %N		; <i32*> [#uses=2]
	store i32 17, i32* %A
	call void @bar( i32* %A )
; CHECK: tail call i32 @foo
	%X = tail call i32 @foo( i32 %N )		; <i32> [#uses=1]
	ret i32 %X
}

