// RUN: %llvmgcc %s -S -emit-llvm -o - | llc

void bork() {
  char Qux[33] = {0};
}
