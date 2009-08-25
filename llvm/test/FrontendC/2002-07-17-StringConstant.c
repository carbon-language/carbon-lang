// RUN: %llvmgcc -S %s -o - | llvm-as -o /dev/null


char * foo() { return "\\begin{"; }
