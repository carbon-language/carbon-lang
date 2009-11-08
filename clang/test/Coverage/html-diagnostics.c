// RUN: rm -rf %t
// RUN: clang-cc -analyze -analyzer-output=html -checker-cfref -o %t %s
// RUN: cat %t/*.html | FileCheck %s

// CHECK: <h3>Annotated Source Code</h3>
// CHECK: Dereference of null pointer

void f0(int x) {
  int *p = &x;

  if (x > 10) {
    if (x == 22)
      p = 0;
  }

  *p = 10;
}


