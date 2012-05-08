// Run lines below; this test is line- and column-sensitive.

void foo(void) __attribute__((availability(macosx,introduced=10.4,deprecated=10.5,obsoleted=10.7), availability(ios,introduced=3.2,deprecated=4.1)));

// RUN: c-index-test -test-load-source all %s | FileCheck %s
// CHECK: (ios, introduced=3.2, deprecated=4.1)  (macosx, introduced=10.4, deprecated=10.5, obsoleted=10.7)
