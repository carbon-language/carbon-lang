; RUN: llvm-as < %s | llc -march=c

; This is a non-normal FP value: it's a nan.
%NAN = global { float } { float 0x7FF8000000000000 } 
%NANs = global { float } { float 0x7FF4000000000000 } 
