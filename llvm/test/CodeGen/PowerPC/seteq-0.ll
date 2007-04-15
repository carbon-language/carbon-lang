; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | \
; RUN:   grep {srwi r., r., 5}

int %eq0(int %a) {
	%tmp.1 = seteq int %a, 0		; <bool> [#uses=1]
	%tmp.2 = cast bool %tmp.1 to int		; <int> [#uses=1]
	ret int %tmp.2
}
