; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | grep 'stw r31, 20(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | grep 'stwu r1, -64(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | grep 'lwz r1, 0(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | grep 'lwz r31, 20(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -disable-fp-elim | grep 'stw r31, 20(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -disable-fp-elim | grep 'stwu r1, -64(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -disable-fp-elim | grep 'lwz r1, 0(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -disable-fp-elim | grep 'lwz r31, 20(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 | grep 'std r31, 40(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 | grep 'stdu r1, -112(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 | grep 'ld r1, 0(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 | grep 'ld r31, 40(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -disable-fp-elim | grep 'std r31, 40(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -disable-fp-elim | grep 'stdu r1, -112(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -disable-fp-elim | grep 'ld r1, 0(r1)' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -disable-fp-elim | grep 'ld r31, 40(r1)'


implementation

int* %f1(uint %n) {
	%tmp = alloca int, uint %n
	ret int* %tmp
}
