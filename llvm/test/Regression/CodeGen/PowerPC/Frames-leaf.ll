; RUN: llvm-as < %s | llc -march=ppc32 | NOT grep 'stw r31, 20(r1)' &&
; RUN: llvm-as < %s | llc -march=ppc32 | NOT grep 'stwu r1, -.*(r1)' &&
; RUN: llvm-as < %s | llc -march=ppc32 | NOT grep 'addi r1, r1, ' &&
; RUN: llvm-as < %s | llc -march=ppc32 | NOT grep 'lwz r31, 20(r1)' &&
; RUN: llvm-as < %s | llc -march=ppc32 -disable-fp-elim | NOT grep 'stw r31, 20(r1)' &&
; RUN: llvm-as < %s | llc -march=ppc32 -disable-fp-elim | NOT grep 'stwu r1, -.*(r1)' &&
; RUN: llvm-as < %s | llc -march=ppc32 -disable-fp-elim | NOT grep 'addi r1, r1, ' &&
; RUN: llvm-as < %s | llc -march=ppc32 -disable-fp-elim | NOT grep 'lwz r31, 20(r1)' &&
; RUN: llvm-as < %s | llc -march=ppc64 | NOT grep 'std r31, 40(r1)' &&
; RUN: llvm-as < %s | llc -march=ppc64 | NOT grep 'stdu r1, -.*(r1)' &&
; RUN: llvm-as < %s | llc -march=ppc64 | NOT grep 'addi r1, r1, ' &&
; RUN: llvm-as < %s | llc -march=ppc64 | NOT grep 'ld r31, 40(r1)' &&
; RUN: llvm-as < %s | llc -march=ppc64 -disable-fp-elim | NOT grep 'stw r31, 40(r1)' &&
; RUN: llvm-as < %s | llc -march=ppc64 -disable-fp-elim | NOT grep 'stdu r1, -.*(r1)' &&
; RUN: llvm-as < %s | llc -march=ppc64 -disable-fp-elim | NOT grep 'addi r1, r1, ' &&
; RUN: llvm-as < %s | llc -march=ppc64 -disable-fp-elim | NOT grep 'ld r31, 40(r1)'


implementation

int* %f1() {
	%tmp = alloca int, uint 2
	ret int* %tmp
}
