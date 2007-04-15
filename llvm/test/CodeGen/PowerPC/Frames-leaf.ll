; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | \
; RUN:   not grep {stw r31, 20(r1)}
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | \
; RUN:   not grep {stwu r1, -.*(r1)}
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | \
; RUN:   not grep {addi r1, r1, }
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | \
; RUN:   not grep {lwz r31, 20(r1)}
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -disable-fp-elim | \
; RUN:   not grep {stw r31, 20(r1)}
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -disable-fp-elim | \
; RUN:   not grep {stwu r1, -.*(r1)}
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -disable-fp-elim | \
; RUN:   not grep {addi r1, r1, }
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -disable-fp-elim | \
; RUN:   not grep {lwz r31, 20(r1)}
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 | \
; RUN:   not grep {std r31, 40(r1)}
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 | \
; RUN:   not grep {stdu r1, -.*(r1)}
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 | \
; RUN:   not grep {addi r1, r1, }
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 | \
; RUN:   not grep {ld r31, 40(r1)}
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -disable-fp-elim | \
; RUN:   not grep {stw r31, 40(r1)}
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -disable-fp-elim | \
; RUN:   not grep {stdu r1, -.*(r1)}
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -disable-fp-elim | \
; RUN:   not grep {addi r1, r1, }
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -disable-fp-elim | \
; RUN:   not grep {ld r31, 40(r1)}


implementation

int* %f1() {
	%tmp = alloca int, uint 2
	ret int* %tmp
}
