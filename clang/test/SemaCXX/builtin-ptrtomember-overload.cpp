// RUN: clang-cc -fsyntax-only -verify %s -std=c++0x

struct A {};

struct B {
	operator A*();
};

struct C : B {

};


void foo(C c, B b, int A::* pmf) {
        // FIXME. Bug or correct? gcc accepts it. It requires derived-to-base followed by user defined conversion to work.
	int j = c->*pmf; // expected-error {{left hand operand to ->* must be a pointer to class compatible with the right hand operand, but is 'struct C'}}
	int i = b->*pmf;
}

