; RUN: llvm-as < %s -o /dev/null -f

; make sure that 'float' values have their value properly truncated.

global float 0x1 
