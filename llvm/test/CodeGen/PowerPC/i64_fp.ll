; fcfid and fctid should be generated when the 64bit feature is enabled, but not
; otherwise.

; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mattr=+64bit | \
; RUN:   grep fcfid
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mattr=+64bit | \
; RUN:   grep fctidz
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mcpu=g5 | \
; RUN:   grep fcfid
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mcpu=g5 | \
; RUN:   grep fctidz
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mattr=-64bit | \
; RUN:   not grep fcfid
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mattr=-64bit | \
; RUN:   not grep fctidz
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mcpu=g4 | \
; RUN:   not grep fcfid
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mcpu=g4 | \
; RUN:   not grep fctidz

double %X(double %Y) {
    %A = cast double %Y to long
    %B = cast long %A to double
	ret double %B
}
