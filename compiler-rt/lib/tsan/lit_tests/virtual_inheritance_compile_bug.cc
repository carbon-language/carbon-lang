// Regression test for http://code.google.com/p/thread-sanitizer/issues/detail?id=3.
// The C++ variant is much more compact that the LLVM IR equivalent.

// RUN: %clangxx_tsan -O1 %s -o %t && %t 2>&1 | FileCheck %s
#include <stdio.h>
struct AAA              {  virtual long aaa () { return 0; } };  // NOLINT
struct BBB: virtual AAA { unsigned long bbb; };  // NOLINT
struct CCC: virtual AAA { };
struct DDD: CCC, BBB { DDD(); };  // NOLINT
DDD::DDD()  { }
int main() {
  DDD d;
  printf("OK\n");
}
// CHECK: OK
