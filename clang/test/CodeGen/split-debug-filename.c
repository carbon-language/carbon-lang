// RUN: %clang -target x86_64-linux-gnu -gsplit-dwarf -S -emit-llvm -o - %s | FileCheck %s
int main (void) {
  return 0;
}

// Testing to ensure that the dwo name gets output into the compile unit.
// CHECK: split-debug-filename.dwo
