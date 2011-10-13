// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s
class C;
class C {
public:
protected:
  typedef int A,B;
  static int sf(), u;

  struct S {};
  enum {}; // expected-warning{{declaration does not declare anything}}
  int; // expected-warning {{declaration does not declare anything}}
  int : 1, : 2;

public:
  void m() {
    int l = 2;
  };

  template<typename T> void mt(T) { };
  ; // expected-warning{{extra ';' inside a class}}

  virtual int vf() const volatile = 0;
  
private:
  int x,f(),y,g();
  inline int h();
  static const int sci = 10;
  mutable int mi;
};
void glo()
{
  struct local {};
}

// PR3177
typedef union {
  __extension__ union {
    int a;
    float b;
  } y;
} bug3177;

// check that we don't consume the token after the access specifier 
// when it's not a colon
class D {
public // expected-error{{expected ':'}}
  int i;
};

// consume the token after the access specifier if it's a semicolon 
// that was meant to be a colon
class E {
public; // expected-error{{expected ':'}}
  int i;
};

// PR11109 must appear at the end of the source file
class pr11109r3 { // expected-note{{to match this '{'}}
  public // expected-error{{expected ':'}} expected-error{{expected '}'}} expected-error{{expected ';' after class}}
