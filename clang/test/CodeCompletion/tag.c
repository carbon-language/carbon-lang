// RUN: clang-cc -fsyntax-only -code-completion-dump=1 %s -o - | FileCheck -check-prefix=CC1 %s &&
// RUN: true

enum X { x };
enum Y { y };
struct Z { };

void X();

void test() {
  enum X { x };
  // CHECK-CC1: X : 0
  // CHECK-CC1: Y : 2
  enum
