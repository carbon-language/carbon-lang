// RUN: %clang_cc1 -fno-math-builtin -emit-llvm -o - %s | FileCheck %s

// Check that the -fno-math-builtin option for -cc1 is working properly,
// by disabling just math builtin generation (other lib functions will
// be generated as builtins).

extern char *p1, *p2;

double pow(double, double);
void *memcpy(void *, const void *, unsigned long);

double foo(double a, double b) {
  memcpy(p1, p2, (unsigned long) b);
// CHECK: call void @llvm.memcpy
  return pow(a, b);
// CHECK: call double @pow
}

