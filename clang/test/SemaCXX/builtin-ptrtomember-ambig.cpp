// RUN: clang-cc -fsyntax-only -verify %s -std=c++0x

struct A {};

struct R {
    operator const A*();
};


struct B  : R {
    operator A*();
};

struct C : B {

};


void foo(C c, int A::* pmf) {
       				// FIXME. Why so many built-in candidates?
	int i = c->*pmf; 	// expected-error {{use of overloaded operator '->*' is ambiguous}} \
				// expected-note {{built-in candidate operator ->* ('struct A const *', 'int const struct A::*')}} \
				// expected-note {{built-in candidate operator ->* ('struct A const *', 'int struct A::*')}} \
				// expected-note {{built-in candidate operator ->* ('struct A *', 'int const struct A::*')}} \
				// expected-note {{built-in candidate operator ->* ('struct A *', 'int struct A::*')}}
}

