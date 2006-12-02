; RUN: llvm-upgrade < %s | llvm-as | llc -march=c

; The C backend was dying when there was no typename for a struct type!

declare int %test(int,{ [32 x int] }*)

