// RUN: clang %s -fsyntax-only -Xclang -verify -fblocks

// clang emits the following warning by default.
// With GCC, -pedantic, -Wreturn-type or -Wall are required to produce the 
// following warning.
int t14() {
  return; // expected-warning {{non-void function 't14' should return a value}}
}

void t15() {
  return 1; // expected-warning {{void function 't15' should not return a value}}
}

int unknown();

void test0() {
}

int test1() {
} // expected-warning {{control reaches end of non-void function}}

int test2() {
  a: goto a;
}

int test3() {
  goto a;
  a: ;
} // expected-warning {{control reaches end of non-void function}}


void halt() {
  a: goto a;
}

void halt2() __attribute__((noreturn));

int test4() {
  halt2();
}

int test5() {
  halt2(), (void)1;
}

int test6() {
  1, halt2();
}

int j;
int unknown_nohalt() {
  return j;
}

int test7() {
  unknown();
} // expected-warning {{control reaches end of non-void function}}

int test8() {
  (void)(1 + unknown());
} // expected-warning {{control reaches end of non-void function}}

int halt3() __attribute__((noreturn));

int test9() {
  (void)(halt3() + unknown());
}

int test10() {
  (void)(unknown() || halt3());
} // expected-warning {{control may reach end of non-void function}}

int test11() {
  (void)(unknown() && halt3());
} // expected-warning {{control may reach end of non-void function}}

int test12() {
  (void)(halt3() || unknown());
}

int test13() {
  (void)(halt3() && unknown());
}

int test14() {
  (void)(1 || unknown());
} // expected-warning {{control reaches end of non-void function}}

int test15() {
  (void)(0 || unknown());
} // expected-warning {{control reaches end of non-void function}}

int test16() {
  (void)(0 && unknown());
} // expected-warning {{control reaches end of non-void function}}

int test17() {
  (void)(1 && unknown());
} // expected-warning {{control reaches end of non-void function}}

int test18() {
  (void)(unknown_nohalt() && halt3());
} // expected-warning {{control may reach end of non-void function}}

int test19() {
  (void)(unknown_nohalt() && unknown());
} // expected-warning {{control reaches end of non-void function}}

int test20() {
  int i;
  if (i)
    return 0;
  else if (0)
    return 2;
} // expected-warning {{control may reach end of non-void function}}

int test21() {
  int i;
  if (i)
    return 0;
  else if (1)
    return 2;
}

int test22() {
  int i;
  switch (i) default: ;
} // expected-warning {{control reaches end of non-void function}}

int test23() {
  int i;
  switch (i) {
  case 0:
    return 0;
  case 2:
    return 2;
  }
} // expected-warning {{control may reach end of non-void function}}

int test24() {
  int i;
  switch (i) {
    case 0:
    return 0;
  case 2:
    return 2;
  default:
    return -1;
  }
}

int test25() {
  1 ? halt3() : unknown();
}

int test26() {
  0 ? halt3() : unknown();
} // expected-warning {{control reaches end of non-void function}}

int j;
void (*fptr)() __attribute__((noreturn));
int test27() {
  switch (j) {
  case 1:
    do { } while (1);
    break;
  case 2:
    for (;;) ;
    break;
  case 3:
    for (;1;) ;
    for (;0;) {
      goto done;
    }
    return 1;
  case 4:    
    while (0) { goto done; }
    return 1;
  case 5:
    while (1) { return 1; }
    break;
  case 6:
    fptr();
    break;
  default:
    return 1;
  }
  done: ;
}

// PR4624
void test28() __attribute__((noreturn));
void test28(x) { while (1) { } }

void exit(int);
int test29() {
  exit(1);
}

#include <setjmp.h>
jmp_buf test30_j;
int test30() {
  if (j)
    longjmp(test30_j, 1);
  else
#if defined(_WIN32) || defined(_WIN64)
    longjmp(test30_j, 2);
#else
    _longjmp(test30_j, 1);
#endif
}

typedef void test31_t(int status);
void test31(test31_t *callback __attribute__((noreturn)));

void test32() {
  ^ (void) { while (1) { } }();
  ^ (void) { if (j) while (1) { } }();
  while (1) { }
}

void test33() {
  if (j) while (1) { }
}
