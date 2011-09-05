// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s

// An aggregate is an array or a class...
struct Aggr {
private:
  static const int n;
  void f();
protected:
  struct Inner { int m; };
public:
  bool &br;
};
bool b;
Aggr ag = { b };

// with no user-provided constructors, ...
struct NonAggr1a {
  NonAggr1a(int, int);
  int k;
};
// In C++0x, 'user-provided' is only defined for special member functions, so
// this type is considered to be an aggregate. This is considered to be
// a language defect.
NonAggr1a na1a = { 42 }; // expected-error {{non-aggregate type 'NonAggr1a'}}

struct NonAggr1b {
  NonAggr1b(const NonAggr1b &);
  int k;
};
NonAggr1b na1b = { 42 }; // expected-error {{non-aggregate type 'NonAggr1b'}}

// no brace-or-equal-initializers for non-static data members, ...
struct NonAggr2 {
  int m = { 123 };
};
NonAggr2 na2 = { 42 }; // expected-error {{non-aggregate type 'NonAggr2'}}

// no private...
struct NonAggr3 {
private:
  int n;
};
NonAggr3 na3 = { 42 }; // expected-error {{non-aggregate type 'NonAggr3'}}

// or protected non-static data members, ...
struct NonAggr4 {
protected:
  int n;
};
NonAggr4 na4 = { 42 }; // expected-error {{non-aggregate type 'NonAggr4'}}

// no base classes, ...
struct NonAggr5 : Aggr {
};
NonAggr5 na5 = { b }; // expected-error {{non-aggregate type 'NonAggr5'}}

// and no virtual functions.
struct NonAggr6 {
  virtual void f();
  int n;
};
NonAggr6 na6 = { 42 }; // expected-error {{non-aggregate type 'NonAggr6'}}
