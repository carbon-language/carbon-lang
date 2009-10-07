// RUN: clang-cc -fsyntax-only -verify %s -std=c++0x

struct A {};

struct B {
	operator A*();
};

struct C : B {

};


void foo(C c, B b, int A::* pmf) {
	int j = c->*pmf; 
	int i = b->*pmf;
}

