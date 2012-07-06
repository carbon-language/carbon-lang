// RUN: %clang_cc1 -analyze -analyzer-checker=core,experimental.core,debug.ExprInspection -analyzer-store=region -analyzer-constraints=basic -analyzer-ipa=inlining -verify %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core,experimental.core,debug.ExprInspection -analyzer-store=region -analyzer-constraints=range -analyzer-ipa=inlining -verify %s

void clang_analyzer_eval(int);

int string_literal_init() {
  char a[] = "abc";
  char b[2] = "abc"; // expected-warning{{too long}}
  char c[5] = "abc";

  clang_analyzer_eval(a[1] == 'b'); // expected-warning{{TRUE}}
  clang_analyzer_eval(b[1] == 'b'); // expected-warning{{TRUE}}
  clang_analyzer_eval(c[1] == 'b'); // expected-warning{{TRUE}}

  clang_analyzer_eval(a[3] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(c[3] == 0); // expected-warning{{TRUE}}

  clang_analyzer_eval(c[4] == 0); // expected-warning{{TRUE}}

  return 42;
}

void nested_compound_literals(int rad) {
  int vec[6][2] = {{0.195, 0.02}, {0.383, 0.067}, {0.55, 0.169},  // expected-warning 6 {{implicit conversion from 'double' to 'int' changes value from}}
                   {0.831, 0.45}, {0.924, 0.617}, {0.98, 0.805}}; // expected-warning 6 {{implicit conversion from 'double' to 'int' changes value from}}
  int a;

  for (a = 0; a < 6; ++a) {
      vec[a][0] *= rad; // no-warning
      vec[a][1] *= rad; // no-warning
  }
}

void nested_compound_literals_float(float rad) {
  float vec[6][2] = {{0.195, 0.02}, {0.383, 0.067}, {0.55, 0.169},
                     {0.831, 0.45}, {0.924, 0.617}, {0.98, 0.805}};
  int a;

  for (a = 0; a < 6; ++a) {
      vec[a][0] *= rad; // no-warning
      vec[a][1] *= rad; // no-warning
  }
}


void struct_as_array() {
  struct simple { int x; int y; };
  struct simple a;
  struct simple *p = &a;

  p->x = 5;
  clang_analyzer_eval(a.x == 5); // expected-warning{{TRUE}}
  clang_analyzer_eval(p[0].x == 5); // expected-warning{{TRUE}}

  p[0].y = 5;
  clang_analyzer_eval(a.y == 5); // expected-warning{{TRUE}}
  clang_analyzer_eval(p->y == 5); // expected-warning{{TRUE}}
}


// PR13264 / <rdar://problem/11802440>
struct point { int x; int y; };
struct circle { struct point o; int r; };
struct circle get_circle() {
  struct circle result;
  result.r = 5;
  result.o = (struct point){0, 0};
  return result;
}

void struct_in_struct() {
  struct circle c;
  c = get_circle();
  // This used to think c.r was undefined because c.o is a LazyCompoundVal.
  clang_analyzer_eval(c.r == 5); // expected-warning{{TRUE}}
}

// We also test with floats because we don't model floats right now,
// and the original bug report used a float.
struct circle_f { struct point o; float r; };
struct circle_f get_circle_f() {
  struct circle_f result;
  result.r = 5.0;
  result.o = (struct point){0, 0};
  return result;
}

float struct_in_struct_f() {
  struct circle_f c;
  c = get_circle_f();

  return c.r; // no-warning
}

