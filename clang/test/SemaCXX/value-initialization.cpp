// RUN: clang-cc -fsyntax-only -verify %s -std=c++0x

struct A {
      ~A();
      const int i;	// expected-note {{declared at}}
};

struct B {
      // B is a non-POD with no user-written constructor.
      // It has a nontrivial generated constructor.
      const int i[12];	// expected-note {{declared at}}
      A a;
};

int main () {
      // Value-initializing a "B" doesn't call the default constructor for
      // "B"; it value-initializes the members of B.  Therefore it shouldn't
      // cause an error on generation of the default constructor for the
      // following:
      new B();	// expected-error {{cannot define the implicit default constructor for 'struct B', because const member 'i'}}
      (void)B();
      (void)A(); // expected-error {{cannot define the implicit default constructor for 'struct A', because const member 'i'}}
}
