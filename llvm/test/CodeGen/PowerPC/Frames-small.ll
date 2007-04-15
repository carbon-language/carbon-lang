; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -o %t1 -f
; RUN  not grep {stw r31, 20(r1)} %t1
; RUN: grep {stwu r1, -16448(r1)} %t1
; RUN: grep {addi r1, r1, 16448} %t1
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN: not grep {lwz r31, 20(r1)}
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim \
; RUN:   -o %t2 -f
; RUN: grep {stw r31, 20(r1)} %t2
; RUN: grep {stwu r1, -16448(r1)} %t2
; RUN: grep {addi r1, r1, 16448} %t2
; RUN: grep {lwz r31, 20(r1)} %t2
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -o %t3 -f
; RUN: not grep {std r31, 40(r1)} %t3
; RUN: grep {stdu r1, -16496(r1)} %t3
; RUN: grep {addi r1, r1, 16496} %t3
; RUN: not grep {ld r31, 40(r1)} %t3
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim \
; RUN:   -o %t4 -f
; RUN: grep {std r31, 40(r1)} %t4
; RUN: grep {stdu r1, -16496(r1)} %t4
; RUN: grep {addi r1, r1, 16496} %t4
; RUN: grep {ld r31, 40(r1)} %t4

implementation

int* %f1() {
	%tmp = alloca int, uint 4095
	ret int* %tmp
}
