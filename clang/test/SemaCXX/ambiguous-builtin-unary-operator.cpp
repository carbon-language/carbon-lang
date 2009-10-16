// RUN: clang-cc -fsyntax-only -verify %s -std=c++0x

struct A {
  operator int&();
  operator long*& ();
};

struct B {
  operator long&();
  operator int*& ();
};

struct C : B, A { };

void test(C c) {
  ++c; // expected-error {{use of overloaded operator '++' is ambiguous}}\
       // expected-note {{built-in candidate operator++(int &)}} \
       // expected-note {{built-in candidate operator++(long &)}} \
       // expected-note {{built-in candidate operator++(long *&)}} \
       // expected-note {{built-in candidate operator++(int *&)}}
}

struct A1 { operator volatile int&(); };

struct B1 { operator volatile long&(); };

struct C1 : B1, A1 { };

void test(C1 c) {
  ++c;	// expected-error {{use of overloaded operator '++' is ambiguous}} \
	// expected-note {{built-in candidate operator++(int volatile &)}} \
	// expected-note {{built-in candidate operator++(long volatile &)}}
}

