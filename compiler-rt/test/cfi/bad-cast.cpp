// RUN: %clangxx_cfi -o %t %s
// RUN: not --crash %t a 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: not --crash %t b 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: not --crash %t c 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %t d 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %t e 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %t f 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: not --crash %t g 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %t h 2>&1 | FileCheck --check-prefix=PASS %s

// RUN: %clangxx_cfi -DB32 -o %t %s
// RUN: not --crash %t a 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: not --crash %t b 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: not --crash %t c 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %t d 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %t e 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %t f 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: not --crash %t g 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %t h 2>&1 | FileCheck --check-prefix=PASS %s

// RUN: %clangxx_cfi -DB64 -o %t %s
// RUN: not --crash %t a 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: not --crash %t b 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: not --crash %t c 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %t d 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %t e 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %t f 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: not --crash %t g 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %t h 2>&1 | FileCheck --check-prefix=PASS %s

// RUN: %clangxx_cfi -DBM -o %t %s
// RUN: not --crash %t a 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: not --crash %t b 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: not --crash %t c 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %t d 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %t e 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %t f 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: not --crash %t g 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: %t h 2>&1 | FileCheck --check-prefix=PASS %s

// RUN: %clangxx_cfi -fsanitize=cfi-cast-strict -o %t %s
// RUN: not --crash %t a 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: not --crash %t b 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: not --crash %t c 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: not --crash %t d 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: not --crash %t e 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: not --crash %t f 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: not --crash %t g 2>&1 | FileCheck --check-prefix=FAIL %s
// RUN: not --crash %t h 2>&1 | FileCheck --check-prefix=FAIL %s

// RUN: %clangxx -o %t %s
// RUN: %t a 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %t b 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %t c 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %t d 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %t e 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %t f 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %t g 2>&1 | FileCheck --check-prefix=PASS %s
// RUN: %t h 2>&1 | FileCheck --check-prefix=PASS %s

// Tests that the CFI enforcement detects bad casts.

#include <stdio.h>
#include <utility>
#include "utils.h"

struct A {
  virtual void f();
};

void A::f() {}

struct B : A {
  virtual void f();
};

void B::f() {}

struct C : A {
};

int main(int argc, char **argv) {
#ifdef B32
  break_optimization(new Deriver<B, 0>);
#endif

#ifdef B64
  break_optimization(new Deriver<B, 0>);
  break_optimization(new Deriver<B, 1>);
#endif

#ifdef BM
  break_optimization(new Deriver<B, 0>);
  break_optimization(new Deriver<B, 1>);
  break_optimization(new Deriver<B, 2>);
#endif

  B *b = new B;
  break_optimization(b);

  // FAIL: 1
  // PASS: 1
  fprintf(stderr, "1\n");

  A a;
  switch (argv[1][0]) {
    case 'a':
      static_cast<B *>(&a); // UB
      break;
    case 'b':
      static_cast<B &>(a); // UB
      break;
    case 'c':
      static_cast<B &&>(a); // UB
      break;
    case 'd':
      static_cast<C *>(&a); // UB, strict only
      break;
    case 'e':
      static_cast<C &>(a); // UB, strict only
      break;
    case 'f':
      static_cast<C &&>(a); // UB, strict only
      break;
    case 'g':
      static_cast<B *>(static_cast<void *>(&a)); // Non-UB bad cast
      break;
    case 'h':
      static_cast<C *>(static_cast<void *>(&a)); // Non-UB bad cast, strict only
      break;
  }

  // FAIL-NOT: 2
  // PASS: 2
  fprintf(stderr, "2\n");
}
