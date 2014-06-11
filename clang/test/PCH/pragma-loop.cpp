// RUN: %clang_cc1 -emit-pch -o %t.a %s
// RUN: %clang_cc1 -include-pch %t.a %s -ast-print -o - | FileCheck %s

// FIXME: A bug in ParsedAttributes causes the order of the attributes to be
// reversed. The checks are consequently in the reverse order below.

// CHECK: #pragma clang loop unroll_count(16)
// CHECK: #pragma clang loop interleave_count(8)
// CHECK: #pragma clang loop vectorize_width(4)
// CHECK: #pragma clang loop unroll(disable)
// CHECK: #pragma clang loop interleave(disable)
// CHECK: #pragma clang loop vectorize(enable)
// CHECK: #pragma clang loop unroll(enable)
// CHECK: #pragma clang loop interleave(enable)
// CHECK: #pragma clang loop vectorize(disable)

#ifndef HEADER
#define HEADER

class pragma_test {
public:
  inline void run1(int *List, int Length) {
    int i = 0;
#pragma clang loop vectorize_width(4)
#pragma clang loop interleave_count(8)
#pragma clang loop unroll_count(16)
    while (i < Length) {
      List[i] = i;
      i++;
    }
  }

  inline void run2(int *List, int Length) {
    int i = 0;
#pragma clang loop vectorize(enable)
#pragma clang loop interleave(disable)
#pragma clang loop unroll(disable)
    while (i - 1 < Length) {
      List[i] = i;
      i++;
    }
  }

  inline void run3(int *List, int Length) {
    int i = 0;
#pragma clang loop vectorize(disable)
#pragma clang loop interleave(enable)
#pragma clang loop unroll(enable)
    while (i - 3 < Length) {
      List[i] = i;
      i++;
    }
  }
};

#else

void test() {
  int List[100];

  pragma_test pt;

  pt.run1(List, 100);
  pt.run2(List, 100);
  pt.run3(List, 100);
}

#endif
