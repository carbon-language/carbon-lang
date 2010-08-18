// RUN: not c-index-test -test-load-source all %s 2> %t.err
// RUN: FileCheck < %t.err -check-prefix=CHECK-LOAD-SOURCE-CRASH %s
// CHECK-LOAD-SOURCE-CRASH: Unable to load translation unit
//
// XFAIL: win32

#pragma clang __debug crash
