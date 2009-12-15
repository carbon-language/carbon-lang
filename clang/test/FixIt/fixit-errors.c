// RUN: %clang_cc1 -pedantic -fixit %s -o - | %clang_cc1 -pedantic -Werror -x c -

/* This is a test of the various code modification hints that are
   provided as part of warning or extension diagnostics. All of the
   warnings will be fixed by -fixit, and the resulting file should
   compile cleanly with -Werror -pedantic. */

struct s; // expected-note{{previous use is here}}

union s *s1; // expected-error{{use of 's' with tag type that does not match previous declaration}}

struct Point {
  float x, y, z;
};

struct Point *get_origin();

void test_point() {
  (void)get_origin->x;
}
