// RUN: %clang_cc1 -fsyntax-only -fobjc-gc -fobjc-nonfragile-abi -verify %s

void f0(__weak id *); // expected-note{{candidate function not viable: 1st argument ('id *') has no lifetime, but parameter has __weak lifetime}}

void test_f0(id *x) {
  f0(x); // expected-error{{no matching function for call to 'f0'}}
}

@interface A
@end

void f1(__weak id*);
void test_f1(__weak A** a) {
  f1(a);
}

@interface B : A
@end

void f2(__weak A**);
void test_f2(__weak B** b) {
  f2(b);
}

