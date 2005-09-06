; fcfid and fctid should be generated when the 64bit feature is enabled, but not 
; otherwise.

; RUN: llvm-as < %s | llc -march=ppc32 -mattr=+64bit | grep 'fcfid' &&
; RUN: llvm-as < %s | llc -march=ppc32 -mattr=+64bit | grep 'fctidz' &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | grep 'fcfid' &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | grep 'fctidz' &&
; RUN: llvm-as < %s | llc -march=ppc32 -mattr=-64bit | not grep 'fcfid' &&
; RUN: llvm-as < %s | llc -march=ppc32 -mattr=-64bit | not grep 'fctidz' &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g4 | not grep 'fcfid' &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g4 | not grep 'fctidz'

double %X(double %Y) {
    %A = cast double %Y to long
    %B = cast long %A to double
	ret double %B
}
