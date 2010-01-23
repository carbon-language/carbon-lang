// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s 

// Verify that we can't initialize non-aggregates with an initializer
// list.
struct NonAggr1 {
  NonAggr1(int) { }

  int m;
};

struct Base { };
struct NonAggr2 : public Base {
  int m;
};

class NonAggr3 {
  int m;
};

struct NonAggr4 {
  int m;
  virtual void f();
};

NonAggr1 na1 = { 17 }; // expected-error{{non-aggregate type 'struct NonAggr1' cannot be initialized with an initializer list}}
NonAggr2 na2 = { 17 }; // expected-error{{non-aggregate type 'struct NonAggr2' cannot be initialized with an initializer list}}
NonAggr3 na3 = { 17 }; // expected-error{{non-aggregate type 'class NonAggr3' cannot be initialized with an initializer list}}
NonAggr4 na4 = { 17 }; // expected-error{{non-aggregate type 'struct NonAggr4' cannot be initialized with an initializer list}}

// PR5817
typedef int type[][2];
const type foo = {0};

// Vector initialization.
typedef short __v4hi __attribute__ ((__vector_size__ (8)));
__v4hi v1 = { (void *)1, 2, 3 }; // expected-error {{cannot initialize a vector element of type 'short' with an rvalue of type 'void *'}}
