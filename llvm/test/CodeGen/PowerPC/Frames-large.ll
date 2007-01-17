; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | not grep 'stw r31, 20(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | grep 'lis r0, -1' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | grep 'ori r0, r0, 32704' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | grep 'stwux r1, r1, r0' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | grep 'lwz r1, 0(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | not grep 'lwz r31, 20(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'stw r31, 20(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'lis r0, -1' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'ori r0, r0, 32704' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'stwux r1, r1, r0' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'lwz r1, 0(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'lwz r31, 20(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 | not grep 'std r31, 40(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 | grep 'lis r0, -1' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 | grep 'ori r0, r0, 32656' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 | grep 'stdux r1, r1, r0' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 | grep 'ld r1, 0(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 | not grep 'ld r31, 40(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'std r31, 40(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'lis r0, -1' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'ori r0, r0, 32656' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'stdux r1, r1, r0' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'ld r1, 0(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'ld r31, 40(r1)'


implementation

int* %f1() {
	%tmp = alloca int, uint 8191
	ret int* %tmp
}
