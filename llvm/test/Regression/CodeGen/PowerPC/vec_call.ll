; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5

<4 x int> %test_arg(<4 x int> %A, <4 x int> %B) {
        %C = add <4 x int> %A, %B
        ret <4 x int> %C
}

<4 x int> %foo() {
	%X = call <4 x int> %test_arg(<4 x int> zeroinitializer, <4 x int> zeroinitializer)
	ret <4 x int> %X
}
