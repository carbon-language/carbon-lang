// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11 -triple=i686-pc-linux-gnu

void ugly_news(int *ip) {
  // These are ill-formed according to one reading of C++98, and at the least
  // have undefined behavior.
  // FIXME: They're ill-formed in C++11.
  (void)new int[-1]; // expected-warning {{array size is negative}}
  (void)new int[2000000000]; // expected-warning {{array is too large}}
}


struct S {
  S(int);
  S();
  ~S();
};

struct T { // expected-note 2 {{not viable}}
  T(int); // expected-note {{not viable}}
};

void fn() {
  (void) new int[2] {1, 2};
  (void) new S[2] {1, 2};
  // C++11 [expr.new]p19:
  //   If the new-expression creates an object or an array of objects of class
  //   type, access and ambiguity control are done for the allocation function,
  //   the deallocation function (12.5), and the constructor (12.1).
  //
  // Note that this happens even if the array bound is constant and the
  // initializer initializes every array element.
  (void) new T[2] {1, 2}; // expected-error {{no matching constructor}} expected-note {{in implicit initialization of array element 2}}
}
