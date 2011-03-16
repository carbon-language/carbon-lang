// RUN: %clang_cc1 -fsyntax-only -verify %s

int ovl(int); // expected-note 3{{candidate function}}
float ovl(float); // expected-note 3{{candidate function}}

template<typename T> T ovl(T); // expected-note 3{{candidate function}}

void test(bool b) {
  (void)((void)0, ovl); // expected-error{{cannot resolve overloaded function 'ovl' from context}}
  // PR7863
  (void)(b? ovl : &ovl); // expected-error{{cannot resolve overloaded function 'ovl' from context}}
  (void)(b? ovl<float> : &ovl); // expected-error{{cannot resolve overloaded function 'ovl' from context}}
  (void)(b? ovl<float> : ovl<float>);
}
