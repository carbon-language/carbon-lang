; Test folding of constantexpr geps into normal geps.
; RUN: llvm-as < %s | opt -instcombine -gcse -instcombine | llvm-dis | not grep getelementptr

%Array = external global [40 x int]

int %test(long %X) {
	%A = getelementptr int* getelementptr ([40 x int]* %Array, long 0, long 0), long %X
	%B = getelementptr [40 x int]* %Array, long 0, long %X
	%a = cast int* %A to int
	%b = cast int* %B to int
	%c = sub int %a, %b
	ret int %c
}
