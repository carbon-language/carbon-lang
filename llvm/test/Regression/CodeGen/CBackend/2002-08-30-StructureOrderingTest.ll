; RUN: llvm-as < %s | llc -march=c

; This testcase fails because the C backend does not arrange to output the 
; contents of a structure type before it outputs the structure type itself.

%Y = uninitialized global { {int } }
%X = uninitialized global { float }
