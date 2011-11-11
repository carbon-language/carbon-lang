void foo() {
  int voodoo;
  voodoo = voodoo + 1;
}

void bar() {
  int dragon;
  dragon = dragon + 1
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
