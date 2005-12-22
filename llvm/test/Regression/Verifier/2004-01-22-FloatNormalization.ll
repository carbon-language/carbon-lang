; RUN: llvm-as < %s -o /dev/null -f
; XFAIL: *

; make sure that invalid 'float' values are caught.

global float 0x1 
