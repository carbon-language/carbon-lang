; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | not grep mr

int %test(int %Y, int %X) {
entry:
	%tmp = tail call int asm "foo $0", "=r"( )		; <int> [#uses=1]
	ret int %tmp
}

int %test2(int %Y, int %X) {
entry:
	%tmp1 = tail call int asm "foo $0, $1", "=r,r"( int %X )		; <int> [#uses=1]
	ret int %tmp1
}
