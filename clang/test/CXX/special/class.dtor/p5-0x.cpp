// RUN: %clang_cc1 -verify -std=c++11 %s

struct NonTrivDtor {
  ~NonTrivDtor();
};
struct DeletedDtor {
  ~DeletedDtor() = delete;
};
class InaccessibleDtor {
  ~InaccessibleDtor() = default;
};

// A defaulted destructor for a class X is defined as deleted if:

// -- X is a union-like class that has a variant member with a non-trivial
// destructor.
union A1 { // expected-note {{here}}
  A1();
  NonTrivDtor n;
};
A1 a1; // expected-error {{deleted function}}
struct A2 { // expected-note {{here}}
  A2();
  union {
    NonTrivDtor n;
  };
};
A2 a2; // expected-error {{deleted function}}
union A3 { // expected-note {{here}}
  A3();
  NonTrivDtor n[3];
};
A3 a3; // expected-error {{deleted function}}
struct A4 { // expected-note {{here}}
  A4();
  union {
    NonTrivDtor n[3];
  };
};
A4 a4; // expected-error {{deleted function}}

// -- any of the non-static data members has class type M (or array thereof) and
// M has a deleted or inaccessible destructor.
struct B1 { // expected-note {{here}}
  B1();
  DeletedDtor a;
};
B1 b1; // expected-error {{deleted function}}
struct B2 { // expected-note {{here}}
  B2();
  InaccessibleDtor a;
};
B2 b2; // expected-error {{deleted function}}
struct B3 { // expected-note {{here}}
  B3();
  DeletedDtor a[4];
};
B3 b3; // expected-error {{deleted function}}
struct B4 { // expected-note {{here}}
  B4();
  InaccessibleDtor a[4];
};
B4 b4; // expected-error {{deleted function}}
union B5 { // expected-note {{here}}
  B5();
  union {
    DeletedDtor a;
  };
};
B5 b5; // expected-error {{deleted function}}
union B6 { // expected-note {{here}}
  B6();
  union {
    InaccessibleDtor a;
  };
};
B6 b6; // expected-error {{deleted function}}

// -- any direct or virtual base class has a deleted or inaccessible destructor.
struct C1 : DeletedDtor { C1(); } c1; // expected-error {{deleted function}} expected-note {{here}}
struct C2 : InaccessibleDtor { C2(); } c2; // expected-error {{deleted function}} expected-note {{here}}
struct C3 : virtual DeletedDtor { C3(); } c3; // expected-error {{deleted function}} expected-note {{here}}
struct C4 : virtual InaccessibleDtor { C4(); } c4; // expected-error {{deleted function}} expected-note {{here}}

// -- for a virtual destructor, lookup of the non-array deallocation function
// results in an ambiguity or a function that is deleted or inaccessible.
class D1 {
  void operator delete(void*);
public:
  virtual ~D1() = default;
} d1; // ok
struct D2 : D1 { // expected-note {{deleted here}}
  // implicitly-virtual destructor
} d2; // expected-error {{deleted function}}
struct D3 {
  virtual ~D3() = default; // expected-note {{deleted here}}
  void operator delete(void*, double = 0.0);
  void operator delete(void*, char = 0);
} d3; // expected-error {{deleted function}}
struct D4 {
  virtual ~D4() = default; // expected-note {{deleted here}}
  void operator delete(void*) = delete;
} d4; // expected-error {{deleted function}}
