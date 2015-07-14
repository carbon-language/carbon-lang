// RUN: %clang_cc1 %s -fapple-kext -fsyntax-only -verify

struct A {
    virtual void f() = 0; // expected-note {{'f' declared here}}
    A() {
        A::f(); // expected-warning {{call to pure virtual member function 'f' has undefined behavior; overrides of 'f' in subclasses are not available in the constructor of 'A'}} // expected-note {{qualified call to 'A'::'f' is treated as a virtual call to 'f' due to -fapple-kext}}
    }
};
