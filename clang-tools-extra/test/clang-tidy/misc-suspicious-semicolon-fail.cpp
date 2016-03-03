// RUN: clang-tidy %s -checks="-*,misc-suspicious-semicolon" -- 2>&1 > %t
// RUN: FileCheck --input-file=%t %s

// Note: This test verifies that, the checker does not emit any warning for
//       files that do not compile.

bool g();

void f() {
  if (g());
  // CHECK-NOT: :[[@LINE-1]]:11: warning: potentially unintended semicolon [misc-suspicious-semicolon]
  int a
}
