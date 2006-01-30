; RUN: llvm-as < %s | llc -march=x86

bool %test(long %X) {
	%B = setlt long %X, 0		; <bool> [#uses=1]
	ret bool %B
}
