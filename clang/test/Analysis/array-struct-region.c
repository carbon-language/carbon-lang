// RUN: %clang_cc1 -analyze -analyzer-checker=core,experimental.core,debug.ExprInspection -analyzer-store=region -analyzer-constraints=basic -verify %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core,experimental.core,debug.ExprInspection -analyzer-store=region -analyzer-constraints=range -verify %s

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

