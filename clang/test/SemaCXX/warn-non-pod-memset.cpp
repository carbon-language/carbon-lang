// RUN: %clang_cc1 -fsyntax-only -verify %s

extern void *memset(void *, int, unsigned);

// Several POD types that should not warn.
struct S1 {} s1;
struct S2 { int x; } s2;
struct S3 { float x, y; S1 s[4]; void (*f)(S1**); } s3;

// Non-POD types that should warn.
struct X1 { X1(); } x1;
struct X2 { ~X2(); } x2;
struct X3 { virtual void f(); } x3;
struct X4 : X2 {} x4;
struct X5 : virtual S1 {} x5;

void test_warn() {
  memset(&x1, 0, sizeof x1); // \
      // expected-warning {{destination for this memset call is a pointer to a non-POD type}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
  memset(&x2, 0, sizeof x2); // \
      // expected-warning {{destination for this memset call is a pointer to a non-POD type}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
  memset(&x3, 0, sizeof x3); // \
      // expected-warning {{destination for this memset call is a pointer to a non-POD type}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
  memset(&x4, 0, sizeof x4); // \
      // expected-warning {{destination for this memset call is a pointer to a non-POD type}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
  memset(&x5, 0, sizeof x5); // \
      // expected-warning {{destination for this memset call is a pointer to a non-POD type}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
}

void test_nowarn() {
  memset(&s1, 0, sizeof s1);
  memset(&s2, 0, sizeof s2);
  memset(&s3, 0, sizeof s3);

  // Unevaluated code shouldn't warn.
  (void)sizeof memset(&x1, 0, sizeof x1);

  // Dead code shouldn't warn.
  if (false) memset(&x1, 0, sizeof x1);
}
