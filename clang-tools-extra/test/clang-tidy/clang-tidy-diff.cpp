// RUN: sed 's/placeholder_for_f/f/' %s > %t.cpp
// RUN: clang-tidy -checks=-*,misc-use-override %t.cpp -- -std=c++11 | FileCheck -check-prefix=CHECK-SANITY %s
// RUN: not diff -u0 %s %t.cpp | python %S/../../clang-tidy/tool/clang-tidy-diff.py -checks=-*,misc-use-override -- -std=c++11 2>&1 | FileCheck %s
struct A {
  virtual void f() {}
  virtual void g() {}
};
// CHECK-NOT: warning
struct B : public A {
  void placeholder_for_f() {}
// CHECK-SANITY: [[@LINE-1]]:8: warning: Use exactly
// CHECK: [[@LINE-2]]:8: warning: Use exactly
  void g() {}
// CHECK-SANITY: [[@LINE-1]]:8: warning: Use exactly
// CHECK-NOT: warning:
};
// CHECK-SANITY-NOT: Suppressed
// CHECK: Suppressed 1 warnings (1 due to line filter).
