// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s
// PR1721

struct s {
  unsigned long long u33: 33;
} a, b;

// This should have %0 and %1 truncated to 33 bits before any operation.
// This can be done using i33 or an explicit and.
_Bool test(void) {
  // CHECK: and i64 %0, 8589934591
  // CHECK: and i64 %1, 8589934591
  return a.u33 + b.u33 != 0;
}
