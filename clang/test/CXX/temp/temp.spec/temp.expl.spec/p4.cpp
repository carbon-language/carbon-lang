// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify %s

struct IntHolder { // expected-note 0-1{{here}} expected-note 2-4{{candidate constructor (the implicit}}
  IntHolder(int); // expected-note 2{{candidate constructor}}
};

template<typename T, typename U>
struct X { // expected-note{{here}}
  void f() { 
    T t; // expected-error{{no matching}}
  }

  void g() { }
  
  struct Inner {
#if __cplusplus >= 201103L
    T value; 	// expected-note {{has no default constructor}}
#else
    // expected-error@-4 {{implicit default}}
    T value; 	// expected-note {{member is declared here}}
#endif
  };
  
  static T value;
};

template<typename T, typename U>
T X<T, U>::value; // expected-error{{no matching constructor}}

IntHolder &test_X_IntHolderInt(X<IntHolder, int> xih) {
  xih.g(); // okay
  xih.f(); // expected-note{{instantiation}}
  
  X<IntHolder, int>::Inner inner;
#if __cplusplus >= 201103L
  // expected-error@-2 {{call to implicitly-deleted}}
#else
  // expected-note@-4 {{first required here}}
#endif
  
  return X<IntHolder, int>::value; // expected-note{{instantiation}}
}

// Explicitly specialize the members of X<IntHolder, long> to not cause
// problems with instantiation.
template<>
void X<IntHolder, long>::f() { }

template<>
struct X<IntHolder, long>::Inner {
  Inner() : value(17) { }
  IntHolder value;
};

template<>
IntHolder X<IntHolder, long>::value = 17;

IntHolder &test_X_IntHolderInt(X<IntHolder, long> xih) {
  xih.g(); // okay
  xih.f(); // okay, uses specialization
  
  X<IntHolder, long>::Inner inner; // okay, uses specialization
  
  return X<IntHolder, long>::value; // okay, uses specialization
}

template<>
X<IntHolder, long>::X() { } // expected-error{{instantiated member}}
