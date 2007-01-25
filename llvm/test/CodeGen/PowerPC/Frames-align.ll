; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | grep 'rlwinm r0, r1, 0, 22, 31' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | grep 'subfic r0, r0, -16448' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 | grep 'rldicl r0, r1, 0, 54'


implementation

int* %f1() {
	%tmp = alloca int, uint 4095, align 1024
	ret int* %tmp
}
