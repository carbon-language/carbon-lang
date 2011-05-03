// RUN: %clang_cc1 -fsyntax-only -Wnon-pod-memaccess -verify %s

extern "C" void *memset(void *, int, unsigned);
extern "C" void *memmove(void *s1, const void *s2, unsigned n);
extern "C" void *memcpy(void *s1, const void *s2, unsigned n);

// Several POD types that should not warn.
struct S1 {} s1;
struct S2 { int x; } s2;
struct S3 { float x, y; S1 s[4]; void (*f)(S1**); } s3;

// We use the C++11 concept of POD for this warning, so ensure a non-aggregate
// still warns.
class C1 {
  int x, y, z;
public:
  void foo() {}
} c1;

// Non-POD types that should warn.
struct X1 { X1(); } x1;
struct X2 { ~X2(); } x2;
struct X3 { virtual void f(); } x3;
struct X4 : X2 {} x4;
struct X5 : virtual S1 {} x5;

void test_warn() {
  memset(&x1, 0, sizeof x1); // \
      // expected-warning {{destination for this 'memset' call is a pointer to non-POD type}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
  memset(&x2, 0, sizeof x2); // \
      // expected-warning {{destination for this 'memset' call is a pointer to non-POD type}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
  memset(&x3, 0, sizeof x3); // \
      // expected-warning {{destination for this 'memset' call is a pointer to dynamic class}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
  memset(&x4, 0, sizeof x4); // \
      // expected-warning {{destination for this 'memset' call is a pointer to non-POD type}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
  memset(&x5, 0, sizeof x5); // \
      // expected-warning {{destination for this 'memset' call is a pointer to dynamic class}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}

  memmove(&x1, 0, sizeof x1); // \
      // expected-warning{{destination for this 'memmove' call is a pointer to non-POD type 'struct X1'}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
  memmove(0, &x1, sizeof x1); // \
      // expected-warning{{source of this 'memmove' call is a pointer to non-POD type 'struct X1'}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
  memcpy(&x1, 0, sizeof x1); // \
      // expected-warning{{destination for this 'memcpy' call is a pointer to non-POD type 'struct X1'}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
  memcpy(0, &x1, sizeof x1); // \
      // expected-warning{{source of this 'memcpy' call is a pointer to non-POD type 'struct X1'}} \
      // expected-note {{explicitly cast the pointer to silence this warning}}
}

void test_nowarn(void *void_ptr) {
  int i, *iptr;
  float y;
  char c;

  memset(&i, 0, sizeof i);
  memset(&iptr, 0, sizeof iptr);
  memset(&y, 0, sizeof y);
  memset(&c, 0, sizeof c);
  memset(void_ptr, 0, 42);
  memset(&s1, 0, sizeof s1);
  memset(&s2, 0, sizeof s2);
  memset(&s3, 0, sizeof s3);
  memset(&c1, 0, sizeof c1);

  // Unevaluated code shouldn't warn.
  (void)sizeof memset(&x1, 0, sizeof x1);

  // Dead code shouldn't warn.
  if (false) memset(&x1, 0, sizeof x1);
}

namespace N {
  void *memset(void *, int, unsigned);
  void test_nowarn() {
    N::memset(&x1, 0, sizeof x1);
  }
}
