// RUN: clang -fsyntax-only -verify %s 
class C {
public:
  auto int errx; // expected-error {{error: storage class specified for a member declaration}}
  register int erry; // expected-error {{error: storage class specified for a member declaration}}
  extern int errz; // expected-error {{error: storage class specified for a member declaration}}

  static void sm() {
    sx = 0;
    this->x = 0; // expected-error {{error: invalid use of 'this' outside of a nonstatic member function}}
    x = 0; // expected-error {{error: invalid use of member 'x' in static member function}}
  }

  class NestedC {
    void m() {
      sx = 0;
      x = 0; // expected-error {{error: invalid use of nonstatic data member 'x'}}
    }
  };

  int b : 1, w : 2;
  int : 1, : 2;
  typedef int E : 1; // expected-error {{error: cannot declare 'E' to be a bit-field type}}
  static int sb : 1; // expected-error {{error: static member 'sb' cannot be a bit-field}}
  static int vs;

  typedef int func();
  func tm;
  func *ptm;
  func btm : 1; // expected-error {{error: bit-field 'btm' with non-integral type}}
  NestedC bc : 1; // expected-error {{error: bit-field 'bc' with non-integral type}}

  enum E { en1, en2 };

  int i = 0; // expected-error {{error: 'i' can only be initialized if it is a static const integral data member}}
  static int si = 0; // expected-error {{error: 'si' can only be initialized if it is a static const integral data member}}
  static const NestedC ci = 0; // expected-error {{error: 'ci' can only be initialized if it is a static const integral data member}}
  static const int nci = vs; // expected-error {{error: initializer element is not a compile-time constant}}
  static const int vi = 0;
  static const E evi = 0;

  void m() {
    sx = 0;
    this->x = 0;
    y = 0;
    this = 0; // expected-error {{error: expression is not assignable}}
  }

  int f1(int p) {
   A z = 6;
   return p + x + this->y + z;
  }

  typedef int A;

private:
  int x,y;
  static int sx;

  static const int number = 50;
  static int arr[number];
};

class C2 {
  void f() {
    static int lx;
    class LC1 {
      int m() { return lx; }
    };
    class LC2 {
      int m() { return lx; }
    };
  }
};
