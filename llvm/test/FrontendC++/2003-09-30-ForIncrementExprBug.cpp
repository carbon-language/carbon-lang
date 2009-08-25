// RUN: %llvmgxx -S %s -o - | llvm-as -o /dev/null

struct C {};

C &foo();

void foox() {
  for (; ; foo());
}

