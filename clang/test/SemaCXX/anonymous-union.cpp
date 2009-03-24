// RUN: clang-cc -fsyntax-only -verify %s
struct X {
  union {
    float f3;
    double d2;
  } named;

  union {
    int i;
    float f;
    
    union {
      float f2;
      mutable double d;
    };
  };

  void test_unqual_references();

  struct {
    int a;
    float b;
  };

  void test_unqual_references_const() const;

  mutable union { // expected-error{{anonymous union at class scope must not have a storage specifier}}
    float c1;
    double c2;
  };
};

void X::test_unqual_references() {
  i = 0;
  f = 0.0;
  f2 = f;
  d = f;
  f3 = 0; // expected-error{{use of undeclared identifier 'f3'}}
  a = 0;
}

void X::test_unqual_references_const() const {
  d = 0.0;
  f2 = 0; // expected-error{{read-only variable is not assignable}}
  a = 0; // expected-error{{read-only variable is not assignable}}
}

void test_unqual_references(X x, const X xc) {
  x.i = 0;
  x.f = 0.0;
  x.f2 = x.f;
  x.d = x.f;
  x.f3 = 0; // expected-error{{no member named 'f3'}}
  x.a = 0;

  xc.d = 0.0;
  xc.f = 0; // expected-error{{read-only variable is not assignable}}
  xc.a = 0; // expected-error{{read-only variable is not assignable}}
}


struct Redecl {
  int x; // expected-note{{previous declaration is here}}
  class y { };

  union {
    int x; // expected-error{{member of anonymous union redeclares 'x'}}
    float y;
    double z; // expected-note{{previous declaration is here}}
    double zz; // expected-note{{previous definition is here}}
  };

  int z; // expected-error{{duplicate member 'z'}}
  void zz(); // expected-error{{redefinition of 'zz' as different kind of symbol}}
};

union { // expected-error{{anonymous unions at namespace or global scope must be declared 'static'}}
  int int_val;
  float float_val;
};

static union {
  int int_val2;
  float float_val2;
};

void f() {
  int_val2 = 0;
  float_val2 = 0.0;
}

void g() {
  union {
    int i;
    float f2;
  };
  i = 0;
  f2 = 0.0;
}

struct BadMembers {
  union {
    struct X { }; // expected-error {{types cannot be declared in an anonymous union}}
    struct { int x; int y; } y;
    
    void f(); // expected-error{{functions cannot be declared in an anonymous union}}
  private: int x1; // expected-error{{anonymous union cannot contain a private data member}}
  protected: float x2; // expected-error{{anonymous union cannot contain a protected data member}}
  };
};

// <rdar://problem/6481130>
typedef union { }; // expected-error{{declaration does not declare anything}}
