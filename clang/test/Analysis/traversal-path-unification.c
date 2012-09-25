// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.DumpTraversal %s | FileCheck %s

int a();
int b();
int c();

void testRemoveDeadBindings() {
  int i = a();
  if (i)
    a();
  else
    b();

  // At this point the symbol bound to 'i' is dead.
  // The effects of a() and b() are identical (they both invalidate globals).
  // We should unify the two paths here and only get one end-of-path node.
  c();
}

// CHECK: --END PATH--
// CHECK-NOT: --END PATH--