enum X { x };
enum Y { y };
struct Z { };

void X();

void test() {
  enum X { x };
  enum
  // RUN: clang-cc -fsyntax-only -code-completion-at=%s:9:7 %s -o - | FileCheck -check-prefix=CC1 %s &&
  // CHECK-CC1: X : 0
  // CHECK-CC1: Y : 2
  // RUN: true
