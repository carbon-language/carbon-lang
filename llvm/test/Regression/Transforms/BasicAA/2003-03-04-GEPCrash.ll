; RUN: llvm-as < %s | opt -basicaa -aa-eval -disable-output
; Test for a bug in BasicAA which caused a crash when querying equality of P1&P2
void %test({[2 x int],[2 x int]}* %A, long %X, long %Y) {
	%P1 = getelementptr {[2 x int],[2 x int]}* %A, long 0, ubyte 0, long %X
	%P2 = getelementptr {[2 x int],[2 x int]}* %A, long 0, ubyte 1, long %Y
	ret void
}
