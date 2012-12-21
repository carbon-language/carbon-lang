// RUN: %clang_cc1 -analyze -analyzer-checker=debug.DumpCallGraph %s -fblocks 2>&1 | FileCheck %s

static void mmm(int y) {
  if (y != 0)
      y++;
  y = y/y;
}

static int foo(int x, int y) {
    mmm(y);
    if (x != 0)
      x++;
    return 5/x;
}

void aaa() {
  foo(1,2);
}

void bbb(int y) {
  int x = (y > 2);
  ^ {
      foo(x, y);
  }();
}

// CHECK:--- Call graph Dump ---
// CHECK: Function: < root > calls: mmm foo aaa < > bbb
// CHECK: Function: bbb calls: < >
// CHECK: Function: < > calls: foo
// CHECK: Function: aaa calls: foo
// CHECK: Function: foo calls: mmm
// CHECK: Function: mmm calls:
