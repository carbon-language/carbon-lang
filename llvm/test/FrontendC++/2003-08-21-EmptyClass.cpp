// RUN: %llvmgxx -S %s -o - | llvm-as -o /dev/null

// This tests compilation of EMPTY_CLASS_EXPR's

struct empty {};

void foo(empty) {}

void bar() { foo(empty()); }
