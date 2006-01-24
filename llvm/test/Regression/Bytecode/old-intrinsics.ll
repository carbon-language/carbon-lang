; RUN: llvm-dis %s.bc-16 -o /dev/null -f &&
; RUN: llc %s.bc-16 -o /dev/null -f -march=c
; Just see if we can disassemble the bc file corresponding to this file.

; XFAIL: *

