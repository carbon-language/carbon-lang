// RUN: %clang -fsanitize=null %s -O3 -o %t
// RUN: %t l 2>&1 | FileCheck %s --check-prefix=CHECK-LOAD
// RUN: %t s 2>&1 | FileCheck %s --check-prefix=CHECK-STORE
// RUN: %t r 2>&1 | FileCheck %s --check-prefix=CHECK-REFERENCE
// RUN: %t m 2>&1 | FileCheck %s --check-prefix=CHECK-MEMBER
// RUN: %t f 2>&1 | FileCheck %s --check-prefix=CHECK-MEMFUN

struct S {
  int f() { return 0; }
  int k;
};

int main(int, char **argv) {
  int *p = 0;
  S *s = 0;

  (void)*p; // ok!

  switch (argv[1][0]) {
  case 'l':
    // CHECK-LOAD: null.cpp:22:12: fatal error: load of null pointer of type 'int'
    return *p;
  case 's':
    // CHECK-STORE: null.cpp:25:5: fatal error: store to null pointer of type 'int'
    *p = 1;
    break;
  case 'r':
    // CHECK-REFERENCE: null.cpp:29:15: fatal error: reference binding to null pointer of type 'int'
    {int &r = *p;}
    break;
  case 'm':
    // CHECK-MEMBER: null.cpp:33:15: fatal error: member access within null pointer of type 'S'
    return s->k;
  case 'f':
    // CHECK-MEMFUN: null.cpp:36:12: fatal error: member call on null pointer of type 'S'
    return s->f();
  }
}
