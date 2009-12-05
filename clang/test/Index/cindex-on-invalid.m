// RUN: not c-index-test -test-load-source local %s > %t 2> %t.err
// RUN: FileCheck %s < %t.err

// CHECK: error: expected identifier or '('
// CHECK: Unable to load translation unit!

int foo;
int
