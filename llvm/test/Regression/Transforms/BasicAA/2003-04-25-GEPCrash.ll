; RUN: llvm-as < %s | opt -basicaa -aa-eval -disable-output
; Test for a bug in BasicAA which caused a crash when querying equality of P1&P2
void %test([17 x ushort]* %mask_bits) {
	%P1 = getelementptr [17 x ushort]* %mask_bits, long 0, long 0
	%P2 = getelementptr [17 x ushort]* %mask_bits, long 252645134, long 0
	ret void
}
