// RUN: %llvmgcc -m32 %s -S -o - | FileCheck %s
// Set alignment on bitfield accesses.

struct S {
  int a, b;
  void *c;
  unsigned d : 8;
  unsigned e : 8;
};

void f0(struct S *a) {
// CHECK: %3 = load i32* %2, align 4
// CHECK: store i32 %4, i32* %2, align 4
  a->e = 0;
}
