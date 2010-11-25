// RUN: %llvmgcc %s -S -o - | llc

void bork() {
  char Qux[33] = {0};
}
