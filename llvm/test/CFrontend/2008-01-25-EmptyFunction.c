// RUN: %llvmgcc -O2 -S -o - -emit-llvm %s | grep llvm.trap
// RUN: %llvmgcc -O2 -S -o - -emit-llvm %s | grep unreachable

void bork() {
  int *address = 0;
  *address = 0;
}
