; RUN: llvm-as < %s | llc -march=x86 | grep rdtsc

declare ulong %llvm.readcyclecounter()

void %foo() {
entry:
%tmp.1 = call ulong %llvm.readcyclecounter ()
ret void
}

