// RUN: clang-cc -fsyntax-only -code-completion-dump=1 %s -o - | FileCheck -check-prefix=CC1 %s &&
// RUN: true
struct Point {
  float x;
  float y;
  float z;
};

void test(struct Point *p) {
  // CHECK-CC1: x
  // CHECK-CC1: y
  // CHECK-CC1: z
  p->