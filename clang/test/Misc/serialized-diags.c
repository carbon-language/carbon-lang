void foo() {
  int voodoo;
  voodoo = voodoo + 1;
}

void bar() {
  int dragon;
  dragon = dragon + 1
}

// Test handling of FixIts that only remove text.
int baz();
void qux(int x) {
  if ((x == baz()))
   return;
}

// RUN: rm -f %t
// RUN: %clang -Wall -fsyntax-only %s --serialize-diagnostics %t 2>&1 /dev/null || true
// RUN: c-index-test -read-diagnostics %t 2>&1 | FileCheck %s

// This test case tests that we can handle multiple diagnostics which contain
// FIXITs at different levels (one at the note, another at the main diagnostic).

// CHECK: {{.*[/\\]}}serialized-diags.c:3:12: warning: variable 'voodoo' is uninitialized when used here [-Wuninitialized]
// CHECK: Range: {{.*[/\\]}}serialized-diags.c:3:12 {{.*[/\\]}}serialized-diags.c:3:18
// CHECK: +-{{.*[/\\]}}serialized-diags.c:2:13: note: initialize the variable 'voodoo' to silence this warning []
// CHECK: +-FIXIT: ({{.*[/\\]}}serialized-diags.c:2:13 - {{.*[/\\]}}serialized-diags.c:2:13): " = 0"
// CHECK: {{.*[/\\]}}serialized-diags.c:8:22: error: expected ';' after expression []
// CHECK: FIXIT: ({{.*[/\\]}}serialized-diags.c:8:22 - {{.*[/\\]}}serialized-diags.c:8:22): ";"
// CHECK: {{.*[/\\]}}serialized-diags.c:14:10: warning: equality comparison with extraneous parentheses [-Wparentheses-equality]
// CHECK: Range: {{.*[/\\]}}serialized-diags.c:14:8 {{.*[/\\]}}serialized-diags.c:14:18
// CHECK: +-{{.*[/\\]}}serialized-diags.c:14:10: note: remove extraneous parentheses around the comparison to silence this warning []
// CHECK: +-FIXIT: ({{.*[/\\]}}serialized-diags.c:14:7 - {{.*[/\\]}}serialized-diags.c:14:8): ""
// CHECK: +-FIXIT: ({{.*[/\\]}}serialized-diags.c:14:18 - {{.*[/\\]}}serialized-diags.c:14:19): ""
// CHECK: +-{{.*[/\\]}}serialized-diags.c:14:10: note: use '=' to turn this equality comparison into an assignment []
// CHECK: +-FIXIT: ({{.*[/\\]}}serialized-diags.c:14:10 - {{.*[/\\]}}serialized-diags.c:14:12): "="
// CHECK: Number of diagnostics: 3

