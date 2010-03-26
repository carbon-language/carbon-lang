// RUN: c-index-test -test-load-source local %s 2>&1 | FileCheck %s

int foo;
int

// CHECK: cindex-on-invalid.m:6:70: error: expected identifier or '('