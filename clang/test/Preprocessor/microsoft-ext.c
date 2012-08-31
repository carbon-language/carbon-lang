// RUN: %clang_cc1 -E -fms-compatibility %s | FileCheck %s

# define M2(x, y) x + y
# define P(x, y) {x, y}
# define M(x, y) M2(x, P(x, y))
M(a, b) // CHECK: a + {a, b}
