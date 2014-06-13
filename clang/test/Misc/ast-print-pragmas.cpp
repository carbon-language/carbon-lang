// RUN: %clang_cc1 -ast-print %s -o - | FileCheck %s

// FIXME: A bug in ParsedAttributes causes the order of the attributes to be
// reversed. The checks are consequently in the reverse order below.

// CHECK: #pragma clang loop interleave_count(8)
// CHECK-NEXT: #pragma clang loop vectorize_width(4)

void test(int *List, int Length) {
  int i = 0;
#pragma clang loop vectorize_width(4)
#pragma clang loop interleave_count(8)
// CHECK-NEXT: while (i < Length)
  while (i < Length) {
    List[i] = i * 2;
    i++;
  }

// CHECK: #pragma clang loop interleave(disable)
// CHECK-NEXT: #pragma clang loop vectorize(enable)

#pragma clang loop vectorize(enable)
#pragma clang loop interleave(disable)
// CHECK-NEXT: while (i - 1 < Length)
  while (i - 1 < Length) {
    List[i] = i * 2;
    i++;
  }

// CHECK: #pragma clang loop interleave(enable)
// CHECK-NEXT: #pragma clang loop vectorize(disable)

#pragma clang loop vectorize(disable)
#pragma clang loop interleave(enable)
// CHECK-NEXT: while (i - 2 < Length)
  while (i - 2 < Length) {
    List[i] = i * 2;
    i++;
  }
}
