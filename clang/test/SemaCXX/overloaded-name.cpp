// RUN: %clang_cc1 -fsyntax-only -verify %s

int ovl(int);
float ovl(float);

template<typename T> T ovl(T);

void test(bool b) {
  (void)((void)0, ovl); // expected-error{{cannot resolve overloaded function from context}}
  // PR7863
  (void)(b? ovl : &ovl); // expected-error{{cannot resolve overloaded function from context}}
  (void)(b? ovl<float> : &ovl); // expected-error{{cannot resolve overloaded function from context}}
  (void)(b? ovl<float> : ovl<float>);
}
