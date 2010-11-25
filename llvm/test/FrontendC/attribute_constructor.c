// RUN: %llvmgcc %s -S -o - | grep llvm.global_ctors

void foo() __attribute__((constructor));
void foo() {
  bar();
}
