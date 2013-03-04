// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -loop-convert %t.cpp -- -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s
// XFAIL: *
#include "structures.h"

void f() {
  // See PR15437 for details.
  PtrSet<int*> int_ptrs;
  for (PtrSet<int*>::iterator I = int_ptrs.begin(),
       E = int_ptrs.end(); I != E; ++I) {
    // CHECK: for (const auto & int_ptr : int_ptrs) {
  }
}
