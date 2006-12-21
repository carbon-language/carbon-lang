; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 | grep -- -56
; PR853

%X = global int* cast (ulong 18446744073709551560 to int*)

