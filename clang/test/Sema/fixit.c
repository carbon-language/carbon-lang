// RUN: clang -fsyntax-only -pedantic -fixit %s -o - | clang-cc -pedantic -Werror -x c -

/* This is a test of the various code modification hints that are
   provided as part of warning or extension diagnostics. Eventually,
   we would like to actually try to perform the suggested
   modifications and compile the result to test that no warnings
   remain. */

void f0(void) { };

struct s {
  int x, y;;
};

_Complex cd;

struct s s0 = { y: 5 };
int array0[5] = { [3] 3 };

void f1(x, y) 
{
}

int i0 = { 17 };

int f2(const char *my_string) {
  // FIXME: terminal output isn't so good when "my_string" is shorter
  // FIXME: Needs an #include hint, too!
  //  return my_string == "foo";
}
