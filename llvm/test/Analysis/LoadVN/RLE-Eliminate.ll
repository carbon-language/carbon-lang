; This testcase ensures that redundant loads are eliminated when they should
; be.  All RL variables (redundant loads) should be eliminated.
;
; RUN: llvm-as < %s | opt -load-vn -gcse | llvm-dis | not grep %RL
;

define i32 @test1(i32* %P) {
	%A = load i32* %P		; <i32> [#uses=1]
	%RL = load i32* %P		; <i32> [#uses=1]
	%C = add i32 %A, %RL		; <i32> [#uses=1]
	ret i32 %C
}

define i32 @test2(i32* %P) {
	%A = load i32* %P		; <i32> [#uses=1]
	br label %BB2

BB2:		; preds = %0
	br label %BB3

BB3:		; preds = %BB2
	%RL = load i32* %P		; <i32> [#uses=1]
	%B = add i32 %A, %RL		; <i32> [#uses=1]
	ret i32 %B
}
