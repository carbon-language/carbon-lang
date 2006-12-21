; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | grep 'stw r31, 20(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | grep 'stwu r1, -64(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | grep 'lwz r1, 0(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | grep 'lwz r31, 20(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'stw r31, 20(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'stwu r1, -64(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'lwz r1, 0(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'lwz r31, 20(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 | grep 'std r31, 40(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 | grep 'stdu r1, -112(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 | grep 'ld r1, 0(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 | grep 'ld r31, 40(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'std r31, 40(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'stdu r1, -112(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'ld r1, 0(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'ld r31, 40(r1)'


implementation

int* %f1(uint %n) {
	%tmp = alloca int, uint %n
	ret int* %tmp
}
