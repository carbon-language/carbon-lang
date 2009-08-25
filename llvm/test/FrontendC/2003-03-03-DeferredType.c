// RUN: %llvmgcc -S %s -o - | llvm-as -o /dev/null




struct foo A;

struct foo {
  int x;
double D;
};

