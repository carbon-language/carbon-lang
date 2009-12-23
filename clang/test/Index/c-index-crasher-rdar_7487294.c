// RUN: c-index-test -test-load-source local %s 2>&1 | FileCheck %s

// This is invalid source.  Previously a double-free caused this
// example to crash c-index-test.

int foo(int x) {
  int y[x * 3];
  help
};

// CHECK: 8:3: error: use of undeclared identifier 'help'
// CHECK:  help
// CHECK: 12:102: error: expected '}'
