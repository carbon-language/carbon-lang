// RUN: cp %s %t
// RUN: %clang_cc1 -pedantic -fixit -x c %t || true
// RUN: grep -v CHECK %t > %t2
// RUN: %clang_cc1 -pedantic -Werror -x c %t
// RUN: FileCheck -input-file=%t2 %t

/* This is a test of the various code modification hints that are
   provided as part of warning or extension diagnostics. All of the
   warnings will be fixed by -fixit, and the resulting file should
   compile cleanly with -Werror -pedantic. */

// FIXME: FIX-IT should add #include <string.h>?
int strcmp(const char *s1, const char *s2);

void f0(void) { };

struct s {
  int x, y;;
};

// CHECK: _Complex double cd;
_Complex cd;

// CHECK: struct s s0 = { .y = 5 };
struct s s0 = { y: 5 };

// CHECK: int array0[5] = { [3] = 3 };
int array0[5] = { [3] 3 };

void f1(x, y)
{
}

int i0 = { 17 };

int test_cond(int y, int fooBar) {
// CHECK: int x = y ? 1 : 4+fooBar;
  int x = y ? 1 4+foobar;
  return x;
}

// CHECK: typedef int int_t;
typedef typedef int int_t;
