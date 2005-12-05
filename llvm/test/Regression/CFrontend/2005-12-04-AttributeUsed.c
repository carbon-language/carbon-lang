// RUN: %llvmgcc %s -S -o - | llvm-as | llvm-dis | grep llvm.used | grep foo | grep X

int X __attribute__((used));
int Y;

void foo() __attribute__((used));

void foo() {}
