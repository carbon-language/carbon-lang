; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm -mattr=+v6
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=arm-apple-darwin -mattr=+v6 |\
; RUN:   grep mov | wc -l | grep 3

int %test(int %x) {
	%tmp = cast int %x to short
	%tmp2 = tail call int %f( int 1, short %tmp )
	ret int %tmp2
}

declare int %f(int, short)
