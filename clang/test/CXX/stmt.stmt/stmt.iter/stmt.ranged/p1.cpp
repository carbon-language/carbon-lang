// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

struct pr12960 {
  int begin;
  void foo(int x) {
    for (int& it : x) { // expected-error {{invalid range expression of type 'int'; no viable 'begin' function available}}
    }
  }
};

namespace std {
  template<typename T>
    auto begin(T &&t) -> decltype(t.begin()) { return t.begin(); } // expected-note 4{{ignored: substitution failure}}
  template<typename T>
    auto end(T &&t) -> decltype(t.end()) { return t.end(); } // expected-note {{candidate template ignored: substitution failure [with T = }}

  template<typename T>
    auto begin(T &&t) -> decltype(t.alt_begin()) { return t.alt_begin(); } // expected-note {{selected 'begin' template [with T = }} \
                                                                              expected-note 4{{candidate template ignored: substitution failure [with T = }}
  template<typename T>
    auto end(T &&t) -> decltype(t.alt_end()) { return t.alt_end(); } // expected-note {{candidate template ignored: substitution failure [with T = }}

  namespace inner {
    // These should never be considered.
    int begin(int);
    int end(int);
  }

  using namespace inner;
}

struct A { // expected-note 2 {{candidate constructor}}
  A();
  int *begin(); // expected-note 3{{selected 'begin' function with iterator type 'int *'}} expected-note {{'begin' declared here}}
  int *end();
};

struct B {
  B();
  int *alt_begin();
  int *alt_end();
};

void f();
void f(int);

void g() {
  for (int a : A())
    A __begin;
  for (char *a : A()) { // expected-error {{cannot initialize a variable of type 'char *' with an lvalue of type 'int'}}
  }
  for (char *a : B()) { // expected-error {{cannot initialize a variable of type 'char *' with an lvalue of type 'int'}}
  }
  // FIXME: Terrible diagnostic here. auto deduction should fail, but does not!
  for (double a : f) { // expected-error {{cannot use type '<overloaded function type>' as a range}}
  }
  for (auto a : A()) {
  }
  for (auto a : B()) {
  }
  for (auto *a : A()) { // expected-error {{variable 'a' with type 'auto *' has incompatible initializer of type 'int'}}
  }
  // : is not a typo for :: here.
  for (A NS:A()) { // expected-error {{no viable conversion from 'int' to 'A'}}
  }
  for (auto not_in_scope : not_in_scope) { // expected-error {{use of undeclared identifier 'not_in_scope'}}
  }

  for (auto a : A())
    for (auto b : A()) {
      __range.begin(); // expected-error {{use of undeclared identifier '__range'}}
      ++__begin; // expected-error {{use of undeclared identifier '__begin'}}
      --__end; // expected-error {{use of undeclared identifier '__end'}}
    }

  for (char c : "test")
    ;
  for (auto a : f()) // expected-error {{cannot use type 'void' as a range}}
    ;

  extern int incomplete[];
  for (auto a : incomplete) // expected-error {{cannot use incomplete type 'int []' as a range}}
    ;
  extern struct Incomplete also_incomplete[2]; // expected-note {{forward declaration}}
  for (auto &a : also_incomplete) // expected-error {{cannot use incomplete type 'struct Incomplete [2]' as a range}}
    ;

  struct VoidBegin {
    void begin(); // expected-note {{selected 'begin' function with iterator type 'void'}}
    void end();
  };
  for (auto a : VoidBegin()) // expected-error {{cannot use type 'void' as an iterator}}
    ;

  struct null_t {
    operator int*();
  };
  struct Differ {
    int *begin(); // expected-note {{selected 'begin' function with iterator type 'int *'}}
    null_t end(); // expected-note {{selected 'end' function with iterator type 'null_t'}}
  };
  for (auto a : Differ()) // expected-error {{'begin' and 'end' must return the same type (got 'int *' and 'null_t')}}
    ;

  for (void f() : "error") // expected-error {{for range declaration must declare a variable}}
    ;

  for (extern int a : A()) {} // expected-error {{loop variable 'a' may not be declared 'extern'}}
  for (static int a : A()) {} // expected-error {{loop variable 'a' may not be declared 'static'}}
  for (register int a : A()) {} // expected-error {{loop variable 'a' may not be declared 'register'}}
  for (constexpr int a : A()) {} // expected-error {{loop variable 'a' may not be declared 'constexpr'}}

  struct NoBeginADL {
    null_t alt_end();
  };
  struct NoEndADL {
    null_t alt_begin();
  };
  for (auto u : NoBeginADL()) { // expected-error {{invalid range expression of type 'NoBeginADL'; no viable 'begin' function available}}
  }
  for (auto u : NoEndADL()) { // expected-error {{invalid range expression of type 'NoEndADL'; no viable 'end' function available}}
  }

  struct NoBegin {
    null_t end();
  };
  struct NoEnd {
    null_t begin();
  };
  for (auto u : NoBegin()) { // expected-error {{range type 'NoBegin' has 'end' member but no 'begin' member}}
  }
  for (auto u : NoEnd()) { // expected-error {{range type 'NoEnd' has 'begin' member but no 'end' member}}
  }

  struct NoIncr {
    void *begin(); // expected-note {{selected 'begin' function with iterator type 'void *'}}
    void *end();
  };
  for (auto u : NoIncr()) { // expected-error {{arithmetic on a pointer to void}}
  }

  struct NoNotEq {
    NoNotEq begin(); // expected-note {{selected 'begin' function with iterator type 'NoNotEq'}}
    NoNotEq end();
    void operator++();
  };
  for (auto u : NoNotEq()) { // expected-error {{invalid operands to binary expression}}
  }

  struct NoCopy {
    NoCopy();
    NoCopy(const NoCopy &) = delete;
    int *begin();
    int *end();
  };
  for (int n : NoCopy()) { // ok
  }

  for (int n : 42) { // expected-error {{invalid range expression of type 'int'; no viable 'begin' function available}}
  }

  for (auto a : *also_incomplete) { // expected-error {{cannot use incomplete type 'struct Incomplete' as a range}}
  }
}

template<typename T, typename U>
void h(T t) {
  for (U u : t) { // expected-error {{no viable conversion from 'A' to 'int'}}
  }
  for (auto u : t) {
  }
}

template void h<A, int>(A);
template void h<A(&)[4], A &>(A(&)[4]);
template void h<A(&)[13], A>(A(&)[13]);
template void h<A(&)[13], int>(A(&)[13]); // expected-note {{requested here}}

template<typename T>
void i(T t) {
  for (auto u : t) { // expected-error {{invalid range expression of type 'A *'; no viable 'begin' function available}} \
                        expected-error {{member function 'begin' not viable}} \
                        expected-note {{when looking up 'begin' function}}

  }
}
template void i<A[13]>(A*); // expected-note {{requested here}}
template void i<const A>(const A); // expected-note {{requested here}}

namespace NS {
  class ADL {};
  int *begin(ADL); // expected-note {{no known conversion from 'NS::NoADL' to 'NS::ADL'}}
  int *end(ADL);

  class NoADL {};
}
int *begin(NS::NoADL);
int *end(NS::NoADL);

struct VoidBeginADL {};
void begin(VoidBeginADL); // expected-note {{selected 'begin' function with iterator type 'void'}}
void end(VoidBeginADL);

void j() {
  for (auto u : NS::ADL()) {
  }
  for (auto u : NS::NoADL()) { // expected-error {{invalid range expression of type 'NS::NoADL'; no viable 'begin' function available}}
  }
  for (auto a : VoidBeginADL()) { // expected-error {{cannot use type 'void' as an iterator}}

  }
}

void example() {
  int array[5] = { 1, 2, 3, 4, 5 };
  for (int &x : array)
    x *= 2;
}
