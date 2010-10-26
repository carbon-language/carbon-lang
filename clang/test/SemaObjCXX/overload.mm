// RUN: %clang_cc1 -fsyntax-only -verify %s
@interface Foo
@end

@implementation Foo

void func(id);

+ zone {
 func(self);
 return self;
}
@end

@protocol P0
@end

@protocol P1
@end

@interface A <P0>
@end

@interface B : A
@end

@interface C <P1>
@end

int& f(A*); // expected-note {{candidate}}
float& f(B*); // expected-note {{candidate}}
void g(A*);

int& h(A*);
float& h(id);

void test0(A* a, B* b, id val) {
  int& i1 = f(a);
  float& f1 = f(b);

  // GCC succeeds here, which is clearly ridiculous.
  float& f2 = f(val); // expected-error {{ambiguous}}

  g(a);
  g(b);
  g(val);
  int& i2 = h(a);
  float& f3 = h(val);
  //  int& i3 = h(b); FIXME: we match GCC here, but shouldn't this work?
}

// We make these errors instead of warnings.  Is that okay?
void test1(A* a) {
  B* b = a; // expected-error{{cannot initialize a variable of type 'B *' with an lvalue of type 'A *'}}
  B *c; c = a; // expected-error{{assigning to 'B *' from incompatible type 'A *'}}
}

void test2(A** ap) {
  B** bp = ap; // expected-warning{{incompatible pointer types converting 'A **' to type 'B **'}}
  bp = ap; // expected-warning{{incompatible pointer types assigning to 'A **' from 'B **'}}
}

// FIXME: we should either allow overloading here or give a better diagnostic
int& cv(A*); // expected-note {{previous declaration}} expected-note 2 {{not viable}}
float& cv(const A*); // expected-error {{cannot be overloaded}}

int& cv2(void*); // expected-note 2 {{candidate}}
float& cv2(const void*); // expected-note 2 {{candidate}}

void cv_test(A* a, B* b, const A* ac, const B* bc) {
  int &i1 = cv(a);
  int &i2 = cv(b);
  float &f1 = cv(ac); // expected-error {{no matching function}}
  float &f2 = cv(bc); // expected-error {{no matching function}}

  // FIXME: these should not be ambiguous
  int& i3 = cv2(a); // expected-error {{ambiguous}}
  float& f3 = cv2(ac); // expected-error {{ambiguous}}
}

// We agree with GCC that these can't be overloaded.
int& qualid(id<P0>); // expected-note {{previous declaration}} expected-note {{not viable}}
float& qualid(id<P1>); // expected-error {{cannot be overloaded}}

void qualid_test(A *a, B *b, C *c) {
  int& i1 = qualid(a);
  int& i2 = qualid(b);

  // This doesn't work only because the overload was rejected above.
  float& f1 = qualid(c); // expected-error {{no matching function}}

  id<P0> p1 = 0;
  p1 = 0;
}


@class NSException;
typedef struct {
    void (*throw_exc)(id);
}
objc_exception_functions_t;

void (*_NSExceptionRaiser(void))(NSException *) {
    objc_exception_functions_t exc_funcs;
    return exc_funcs.throw_exc; // expected-warning{{incompatible pointer types converting 'void (*)(id)' to type 'void (*)(NSException *)'}}
}

namespace test5 {
  void foo(bool);
  void foo(void *);

  void test(id p) {
    foo(p);
  }
}

// rdar://problem/8592139
// FIXME: this should resolve to the unavailable candidate
namespace test6 {
  void foo(id); // expected-note {{candidate}}
  void foo(A*) __attribute__((unavailable)); // expected-note {{explicitly made unavailable}}

  void test(B *b) {
    foo(b); // expected-error {{call to 'foo' is ambiguous}}
  }
}
