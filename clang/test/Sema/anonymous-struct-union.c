// RUN: clang -fsyntax-only -verify %s
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
      double d;
    };
  };

  struct {
    int a;
    float b;
  };
};

void test_unqual_references(struct X x, const struct X xc) {
  x.i = 0;
  x.f = 0.0;
  x.f2 = x.f;
  x.d = x.f;
  x.f3 = 0; // expected-error{{no member named 'f3'}}
  x.a = 0;

  xc.d = 0.0; // expected-error{{read-only variable is not assignable}}
  xc.f = 0; // expected-error{{read-only variable is not assignable}}
  xc.a = 0; // expected-error{{read-only variable is not assignable}}
}


struct Redecl {
  int x; // expected-note{{previous declaration is here}}
  struct y { };

  union {
    int x; // expected-error{{member of anonymous union redeclares 'x'}}
    float y;
    double z; // expected-note{{previous declaration is here}}
    double zz; // expected-note{{previous declaration is here}}
  };

  int z; // expected-error{{duplicate member 'z'}}
  void zz(); // expected-error{{duplicate member 'zz'}} \
            //  expected-error{{field 'zz' declared as a function}}
};

union { // expected-error{{anonymous unions must be struct or union members}}
  int int_val;
  float float_val;
};

static union { // expected-error{{anonymous unions must be struct or union members}}
  int int_val2;
  float float_val2;
};

void f() {
  int_val2 = 0; // expected-error{{use of undeclared identifier}}
  float_val2 = 0.0; // expected-error{{use of undeclared identifier}}
}

void g() {
  union { // expected-error{{anonymous unions must be struct or union members}}
    int i;
    float f2;
  };
  i = 0; // expected-error{{use of undeclared identifier}}
  f2 = 0.0; // expected-error{{use of undeclared identifier}}
}

// <rdar://problem/6483159>
struct s0 { union { int f0; }; };

// <rdar://problem/6481130>
typedef struct { }; // expected-error{{declaration does not declare anything}}
