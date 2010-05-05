// RUN: %llvmgcc %s -S -o - | FileCheck %s
struct __attribute((packed)) x {int a : 24;};
int a(struct x* g) {
  // CHECK: load i24
  return g->a;
}
