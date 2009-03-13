; RUN: llvm-as < %s | llc | not grep cmov

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"

; Should compile to setcc | -2.
; rdar://6668608
define i32 @test(i32* nocapture %P) nounwind readonly {
entry:
	%0 = load i32* %P, align 4		; <i32> [#uses=1]
	%1 = icmp sgt i32 %0, 41		; <i1> [#uses=1]
	%iftmp.0.0 = select i1 %1, i32 -1, i32 -2		; <i32> [#uses=1]
	ret i32 %iftmp.0.0
}

; 	setl	%al
;	movzbl	%al, %eax
;	leal	4(%eax,%eax,8), %eax
define i32 @test2(i32* nocapture %P) nounwind readonly {
entry:
	%0 = load i32* %P, align 4		; <i32> [#uses=1]
	%1 = icmp sgt i32 %0, 41		; <i1> [#uses=1]
	%iftmp.0.0 = select i1 %1, i32 4, i32 13		; <i32> [#uses=1]
	ret i32 %iftmp.0.0
}

