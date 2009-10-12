// RUN: clang-cc -fsyntax-only -verify %s -std=c++0x

struct A {};
struct E {};

struct R {
    operator A*();
    operator E*();	// expected-note{{candidate function}}
};


struct S {
    operator A*();
    operator E*();	// expected-note{{candidate function}}
};

struct B  : R {
    operator A*();
};

struct C : B {

};

void foo(C c, int A::* pmf) {
	int i = c->*pmf; 
}

struct B1  : R, S {
    operator A*();
};

struct C1 : B1 {

};

void foo1(C1 c1, int A::* pmf) {
        int i = c1->*pmf;
        c1->*pmf = 10;
}

void foo1(C1 c1, int E::* pmf) {
        // FIXME. Error reporting needs much improvement here.
        int i = c1->*pmf;	// expected-error {{left hand operand to ->* must be a pointer to class compatible with the right hand operand, but is 'struct C1'}} \
                                // expected-note {{because of ambiguity in conversion of 'struct C1' to 'struct E *'}}
}
