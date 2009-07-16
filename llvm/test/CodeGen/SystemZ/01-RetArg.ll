; RUN: llvm-as < %s | llc -march=systemz

define i64 @foo(i64 %a, i64 %b) {
entry:
    ret i64 %b
}