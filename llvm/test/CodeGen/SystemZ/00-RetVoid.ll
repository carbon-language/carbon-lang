; RUN: llvm-as < %s | llc -march=systemz

define void @foo() {
entry:
    ret void
}