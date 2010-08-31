; RUN: opt < %s -tailcallelim -S | FileCheck %s

define i32 @test1_factorial(i32 %x) {
entry:
	%tmp.1 = icmp sgt i32 %x, 0		; <i1> [#uses=1]
	br i1 %tmp.1, label %then, label %else
then:		; preds = %entry
	%tmp.6 = add i32 %x, -1		; <i32> [#uses=1]
	%tmp.4 = call i32 @test1_factorial( i32 %tmp.6 )		; <i32> [#uses=1]
	%tmp.7 = mul i32 %tmp.4, %x		; <i32> [#uses=1]
	ret i32 %tmp.7
else:		; preds = %entry
	ret i32 1
}

; CHECK: define i32 @test1_factorial
; CHECK: phi i32
; CHECK-NOT: call i32
; CHECK: else:

; This is a more aggressive form of accumulator recursion insertion, which 
; requires noticing that X doesn't change as we perform the tailcall.

define i32 @test2_mul(i32 %x, i32 %y) {
entry:
	%tmp.1 = icmp eq i32 %y, 0		; <i1> [#uses=1]
	br i1 %tmp.1, label %return, label %endif
endif:		; preds = %entry
	%tmp.8 = add i32 %y, -1		; <i32> [#uses=1]
	%tmp.5 = call i32 @test2_mul( i32 %x, i32 %tmp.8 )		; <i32> [#uses=1]
	%tmp.9 = add i32 %tmp.5, %x		; <i32> [#uses=1]
	ret i32 %tmp.9
return:		; preds = %entry
	ret i32 %x
}

; CHECK: define i32 @test2_mul
; CHECK: phi i32
; CHECK-NOT: call i32
; CHECK: return: