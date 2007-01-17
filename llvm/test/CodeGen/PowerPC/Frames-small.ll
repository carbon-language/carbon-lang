; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | not grep 'stw r31, 20(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | grep 'stwu r1, -16448(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | grep 'addi r1, r1, 16448' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | not grep 'lwz r31, 20(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'stw r31, 20(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'stwu r1, -16448(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'addi r1, r1, 16448' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'lwz r31, 20(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 | not grep 'std r31, 40(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 | grep 'stdu r1, -16496(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 | grep 'addi r1, r1, 16496' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 | not grep 'ld r31, 40(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'std r31, 40(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'stdu r1, -16496(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'addi r1, r1, 16496' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | grep 'ld r31, 40(r1)'


implementation

int* %f1() {
	%tmp = alloca int, uint 4095
	ret int* %tmp
}
