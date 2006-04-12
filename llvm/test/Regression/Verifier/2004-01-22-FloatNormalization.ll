; RUN: not llvm-as < %s -o /dev/null -f
; XFAIL: 3.4 

; make sure that invalid 'float' values are caught.

global float 0x1 
