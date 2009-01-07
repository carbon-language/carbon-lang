; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis | not grep alloca
; PR3290

;; Store of integer to whole alloca struct.
define i32 @test1(i64 %V) nounwind {
	%X = alloca {{i32, i32}}
	%Y = bitcast {{i32,i32}}* %X to i64*
	store i64 %V, i64* %Y

	%A = getelementptr {{i32,i32}}* %X, i32 0, i32 0, i32 0
	%B = getelementptr {{i32,i32}}* %X, i32 0, i32 0, i32 1
	%a = load i32* %A
	%b = load i32* %B
	%c = add i32 %a, %b
	ret i32 %c
}

;; Store of integer to whole struct/array alloca.
define float @test2(i128 %V) nounwind {
	%X = alloca {[4 x float]}
	%Y = bitcast {[4 x float]}* %X to i128*
	store i128 %V, i128* %Y

	%A = getelementptr {[4 x float]}* %X, i32 0, i32 0, i32 0
	%B = getelementptr {[4 x float]}* %X, i32 0, i32 0, i32 3
	%a = load float* %A
	%b = load float* %B
	%c = add float %a, %b
	ret float %c
}

