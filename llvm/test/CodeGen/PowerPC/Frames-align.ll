; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | \
; RUN:   grep {rlwinm r0, r1, 0, 22, 31}
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | \
; RUN:   grep {subfic r0, r0, -16448}
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc64 -mtriple=powerpc-apple-darwin8 | \
; RUN:   grep {rldicl r0, r1, 0, 54}

implementation

int* %f1() {
	%tmp = alloca int, uint 4095, align 1024
	ret int* %tmp
}
