; RUN: llvm-upgrade < %s | llvm-as | opt -tailcallelim | llvm-dis > %t
; RUN: grep sub %t
; RUN: not grep call %t

int %test(int %X) {
	%Y = sub int %X, 1
	%Z = call int %test(int %Y)
	ret int undef
}
