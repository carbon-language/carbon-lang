; RUN: not llvm-as %s |& grep error

void %foo() {
