// RUN: %clang_cc1 -fsyntax-only -Winconsistent-missing-override -verify -std=c++11 %s
struct A
{
    virtual void foo();
    virtual void bar(); // expected-note {{overridden virtual function is here}}
    virtual void gorf() {}
    virtual void g() = 0; // expected-note {{overridden virtual function is here}}
};
 
struct B : A
{
    void foo() override;
    void bar(); // expected-warning {{'bar' overrides a member function but is not marked 'override'}}
};

struct C : B 
{
    virtual void g() = 0;  // expected-warning {{'g' overrides a member function but is not marked 'override'}}
    virtual void gorf() override {}
};

