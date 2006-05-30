; RUN: llvm-as < %s | llc -march=x86
; RUN: llvm-as < %s | llc -march=x86 | not grep orl
int %test(int %x) {
	%tmp1 = shl int %x, ubyte 3
	%tmp2 = add int %tmp1, 7
	ret int %tmp2
}
