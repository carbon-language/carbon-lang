; RUN: llvm-as < %s | opt -load-vn -gcse -instcombine | \
; RUN:    llvm-dis | not grep load

@X = constant [2 x i32] [i32 4, i32 5]

define i32 @test(i32* %Y, i64 %idx) {
    %P = getelementptr [2 x i32]* @X, i64 0, i64 %idx
	%A = load i32* %P      ; Load from invariant memory
	store i32 4, i32* %Y   ; Store could not be to @X
	%B = load i32* %P
	%C = sub i32 %A, %B
	ret i32 %C
}
