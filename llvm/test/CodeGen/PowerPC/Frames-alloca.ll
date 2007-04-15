; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | \
; RUN:   grep {stw r31, 20(r1)}
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | \
; RUN:   grep {stwu r1, -64(r1)}
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | grep {lwz r1, 0(r1)}
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | \
; RUN:   grep {lwz r31, 20(r1)}
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | \
; RUN:   grep {stw r31, 20(r1)}
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | \
; RUN:   grep {stwu r1, -64(r1)}
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | \
; RUN:   grep {lwz r1, 0(r1)}
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | \
; RUN:   grep {lwz r31, 20(r1)}
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc64 -mtriple=powerpc-apple-darwin8 | \
; RUN:   grep {std r31, 40(r1)}
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc64 -mtriple=powerpc-apple-darwin8 | \
; RUN:   grep {stdu r1, -112(r1)}
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc64 -mtriple=powerpc-apple-darwin8 | \
; RUN:   grep {ld r1, 0(r1)}
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc64 -mtriple=powerpc-apple-darwin8 | \
; RUN:   grep {ld r31, 40(r1)}
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | \
; RUN:   grep {std r31, 40(r1)}
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | \
; RUN:   grep {stdu r1, -112(r1)}
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | \
; RUN:   grep {ld r1, 0(r1)}
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | \
; RUN:   grep {ld r31, 40(r1)}


implementation

int* %f1(uint %n) {
	%tmp = alloca int, uint %n
	ret int* %tmp
}
