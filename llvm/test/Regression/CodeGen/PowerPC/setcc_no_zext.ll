; RUN: llvm-as < %s | llc -march=ppc32 | not grep rlwinm

int %setcc_one_or_zero(int* %a) {
entry:
	%tmp.1 = setne int* %a, null
	%inc.1 = cast bool %tmp.1 to int
	ret int %inc.1
}
