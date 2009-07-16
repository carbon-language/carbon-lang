; RUN: llvm-as < %s | llc -march=systemz

define i64 @foo() {
entry:
    ret i64 0
}