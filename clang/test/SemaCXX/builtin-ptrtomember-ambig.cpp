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
				// expected-note 40 {{built-in candidate operator ->* ('struct A}}
}

