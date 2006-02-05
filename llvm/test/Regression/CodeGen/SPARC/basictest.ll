; RUN: llvm-as < %s | llc -march=sparc

int %test(int %X) {
	%tmp.1 = add int %X, 1		; <int> [#uses=1]
	ret int %tmp.1
}
