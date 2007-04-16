; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 | grep rdtsc
; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86-64 | grep rdtsc

declare ulong %llvm.readcyclecounter()

ulong %foo() {
%tmp.1 = call ulong %llvm.readcyclecounter ()
ret ulong %tmp.1
}

