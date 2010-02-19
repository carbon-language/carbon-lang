// RUN: not c-index-test -test-load-source local %s > %t 2> %t.err
// RUN: FileCheck %s < %t.err
// XFAIL: win32
// CHECK: error: expected identifier or '('
// CHECK: Unable to load translation unit!

int foo;
int
