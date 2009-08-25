// RUN: %llvmgxx -S %s -o - | llvm-as -o /dev/null

class Empty {};

void foo(Empty E);

void bar() {
  foo(Empty());
}

