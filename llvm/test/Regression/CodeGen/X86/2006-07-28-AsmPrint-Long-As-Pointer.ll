; RUN: llvm-as < %s | llc -march=x86 | grep 4294967240
; PR853

%X = global int* cast (ulong 18446744073709551560 to int*)

