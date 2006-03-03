; RUN: llvm-as < %s | llc -march=x86 -stats 2>&1 | grep 'asm-printer' | grep 7

int %g(int %a, int %b) {
	%tmp.1 = shl int %b, ubyte 1
	%tmp.3 = add int %tmp.1, %a
	%tmp.5 = mul int %tmp.3, %a
	%tmp.8 = mul int %b, %b
	%tmp.9 = add int %tmp.5, %tmp.8
	ret int %tmp.9
}
