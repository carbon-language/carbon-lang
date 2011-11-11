void foo() {
  int voodoo;
  voodoo = voodoo + 1;
}

// RUN: %clang -Wall -fsyntax-only %s --serialize-diagnostics %t
// RUN: c-index-test -read-diagnostics %t 2>&1 | FileCheck %s
// RUN: rm -f %t

// NOTE: it is important that this test case only contain a single issue.  This test case checks
// if we can handle serialized diagnostics that contain only one diagnostic.

// CHECK: {{.*}}serialized-diags-single-issue.c:3:12: warning: variable 'voodoo' is uninitialized when used here [-Wuninitialized]
// CHECK: Range: {{.*}}serialized-diags-single-issue.c:3:12 {{.*}}serialized-diags-single-issue.c:3:18
// CHECK: +-{{.*}}serialized-diags-single-issue.c:2:13: note: initialize the variable 'voodoo' to silence this warning []
// CHECK: +-FIXIT: {{.*}}serialized-diags-single-issue.c:2:13 - {{.*}}serialized-diags-single-issue.c:2:13): " = 0"