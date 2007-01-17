; RUN: llvm-upgrade < %s | llvm-as -o /dev/null -f

; The old C front-end never generated empty structures, now the new one
; can.  For some reason we never handled them in the parser. Weird.

%X = global {} {}
