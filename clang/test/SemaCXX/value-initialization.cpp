// RUN: clang-cc -fsyntax-only -verify %s -std=c++0x

struct A {
     const int i;	// expected-note {{declared at}}
     virtual void f() { } 
};

int main () {
      (void)A();	// expected-error {{cannot define the implicit default constructor for 'struct A', because const member 'i' cannot be default-initialized}}
}
