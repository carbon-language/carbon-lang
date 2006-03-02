; RUN: llvm-as < %s | llc -march=x86 | not grep 'subl.*%esp'

int %f(int %a, int %b) {
	%tmp.2 = mul int %a, %a
	%tmp.5 = shl int %a, ubyte 1
	%tmp.6 = mul int %tmp.5, %b
	%tmp.10 = mul int %b, %b
	%tmp.7 = add int %tmp.10, %tmp.2
	%tmp.11 = add int %tmp.7, %tmp.6
	ret int %tmp.11
}
