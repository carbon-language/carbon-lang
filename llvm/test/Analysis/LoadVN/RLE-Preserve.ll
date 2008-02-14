; This testcase ensures that redundant loads are preserved when they are not
; allowed to be eliminated.
; RUN: llvm-as < %s | opt -load-vn -gcse | llvm-dis | grep sub
;

define i32 @test1(i32* %P) {
	%A = load i32* %P		; <i32> [#uses=1]
	store i32 1, i32* %P
	%B = load i32* %P		; <i32> [#uses=1]
	%C = sub i32 %A, %B		; <i32> [#uses=1]
	ret i32 %C
}

define i32 @test2(i32* %P) {
	%A = load i32* %P		; <i32> [#uses=1]
	br label %BB2

BB2:		; preds = %0
	store i32 5, i32* %P
	br label %BB3

BB3:		; preds = %BB2
	%B = load i32* %P		; <i32> [#uses=1]
	%C = sub i32 %A, %B		; <i32> [#uses=1]
	ret i32 %C
}
