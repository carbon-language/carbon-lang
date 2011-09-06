// RUN: cp %s %t
// RUN: not %clang_cc1 -pedantic -Wunused-label -fixit -x c %t
// RUN: grep -v CHECK %t > %t2
// RUN: %clang_cc1 -pedantic -Wunused-label -Werror -x c %t
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

#define ONE 1
#define TWO 2

int test_cond(int y, int fooBar) {
// CHECK: int x = y ? 1 : 4+fooBar;
  int x = y ? 1 4+foobar;
// CHECK: x = y ? ONE : TWO;
  x = y ? ONE TWO;
  return x;
}

// CHECK: typedef int int_t;
typedef typedef int int_t;

// <rdar://problem/7159693>
enum Color { 
  Red // expected-error{{missing ',' between enumerators}}
  Green = 17 // expected-error{{missing ',' between enumerators}}
  Blue,
};

// rdar://9295072
struct test_struct {
  // CHECK: struct test_struct *struct_ptr;
  test_struct *struct_ptr; // expected-error {{must use 'struct' tag to refer to type 'test_struct'}}
};

void removeUnusedLabels(char c) {
  L0 /*removed comment*/:        c++;
  removeUnusedLabels(c);
  L1:
  c++;
  /*preserved comment*/ L2  :        c++;
  LL
  : c++;
  c = c + 3; L4: return;
}
