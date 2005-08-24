; RUN: llvm-as < %s | llc -march=ppc32 | grep 'srwi r., r., 5'

int %eq0(int %a) {
	%tmp.1 = seteq int %a, 0		; <bool> [#uses=1]
	%tmp.2 = cast bool %tmp.1 to int		; <int> [#uses=1]
	ret int %tmp.2
}
