// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s
//
// CHECK: alloca %struct.MemsetRange, align 16

struct MemsetRange {
  int Start, End;
  unsigned Alignment;
  int TheStores __attribute__((aligned(16)));
};
void foobar() {
  (void) MemsetRange();
}
