// RUN: clang-cc -pedantic -fixit %s -o - | grep -v 'CHECK' > %t
// RUN: clang-cc -pedantic -Werror -x c -
// RUN: FileCheck -input-file=%t %s

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

int f2(const char *my_string) {
  // FIXME: terminal output isn't so good when "my_string" is shorter
// CHECK: return strcmp(my_string , "foo") == 0;
  return my_string == "foo";
}

