; RUN: llvm-as < %s | llc -march=alpha | grep rpcc

declare ulong %llvm.readcyclecounter()

ulong %foo() {
entry:
%tmp.1 = call ulong %llvm.readcyclecounter ()
ret ulong %tmp.1
}

