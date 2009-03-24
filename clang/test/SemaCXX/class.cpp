// RUN: clang-cc -fsyntax-only -verify %s 
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
  typedef int E : 1; // expected-error {{typedef member 'E' cannot be a bit-field}}
  static int sb : 1; // expected-error {{error: static member 'sb' cannot be a bit-field}}
  static int vs;

  typedef int func();
  func tm;
  func *ptm;
  func btm : 1; // expected-error {{bit-field 'btm' has non-integral type}}
  NestedC bc : 1; // expected-error {{bit-field 'bc' has non-integral type}}

  enum E1 { en1, en2 };

  int i = 0; // expected-error {{error: 'i' can only be initialized if it is a static const integral data member}}
  static int si = 0; // expected-error {{error: 'si' can only be initialized if it is a static const integral data member}}
  static const NestedC ci = 0; // expected-error {{error: 'ci' can only be initialized if it is a static const integral data member}}
  static const int nci = vs; // expected-error {{in-class initializer is not an integral constant expression}}
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

  virtual int viv; // expected-error {{'virtual' can only appear on non-static member functions}}
  virtual static int vsif(); // expected-error {{error: 'virtual' can only appear on non-static member functions}}
  virtual int vif();

private:
  int x,y;
  static int sx;

  mutable int mi;
  mutable int &mir; // expected-error {{error: 'mutable' cannot be applied to references}}
  mutable void mfn(); // expected-error {{error: 'mutable' cannot be applied to functions}}
  mutable const int mci; // expected-error {{error: 'mutable' and 'const' cannot be mixed}}

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

struct C3 {
  int i;
  mutable int j;
};
void f()
{
  const C3 c3 = { 1, 2 };
  (void)static_cast<int*>(&c3.i); // expected-error {{static_cast from 'int const *' to 'int *' is not allowed}}
  // but no error here
  (void)static_cast<int*>(&c3.j);
}

// Play with mutable a bit more, to make sure it doesn't crash anything.
mutable int gi; // expected-error {{error: 'mutable' can only be applied to member variables}}
mutable void gfn(); // expected-error {{illegal storage class on function}}
void ogfn()
{
  mutable int ml; // expected-error {{error: 'mutable' can only be applied to member variables}}

  // PR3020: This used to crash due to double ownership of C4.
  struct C4;
  C4; // expected-error {{declaration does not declare anything}}
}

struct C4 {
  void f(); // expected-note{{previous declaration is here}}
  int f; // expected-error{{duplicate member 'f'}}
};
