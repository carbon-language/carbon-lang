// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

namespace std { class type_info; };

namespace ExplicitCapture {
  class C {
    int Member;

    static void Overload(int);
    void Overload();
    virtual C& Overload(float);

    void ImplicitThisCapture() {
      [](){(void)Member;}; // expected-error {{'this' cannot be implicitly captured in this context}} expected-error {{not supported yet}}
      [&](){(void)Member;}; // expected-error {{not supported yet}}
      // 'this' captures below don't actually work yet
      [this](){(void)Member;}; // expected-error{{lambda expressions are not supported yet}}
      [this]{[this]{};}; // expected-error 2{{lambda expressions are not supported yet}}
      []{[this]{};};// expected-error {{'this' cannot be implicitly captured in this context}} expected-error 2 {{not supported yet}}
      []{Overload(3);}; // expected-error {{not supported yet}}
      []{Overload();}; // expected-error {{'this' cannot be implicitly captured in this context}} expected-error {{not supported yet}}
      []{(void)typeid(Overload());};// expected-error {{not supported yet}}
      []{(void)typeid(Overload(.5f));};// expected-error {{'this' cannot be implicitly captured in this context}} expected-error {{not supported yet}}
    }
  };

  void f() {
    [this] () {}; // expected-error {{'this' cannot be captured in this context}} expected-error {{not supported yet}}
  }
}

namespace ReturnDeduction {
  void test() {
    [](){ return 1; }; // expected-error {{not supported yet}}
    [](){ return 1; }; // expected-error {{not supported yet}}
    [](){ return ({return 1; 1;}); }; // expected-error {{not supported yet}}
    [](){ return ({return 'c'; 1;}); }; // expected-error {{not supported yet}} expected-error {{must match previous return type}}
    []()->int{ return 'c'; return 1; }; // expected-error {{not supported yet}}
    [](){ return 'c'; return 1; }; // expected-error {{not supported yet}} expected-error {{must match previous return type}}
    []() { return; return (void)0; }; // expected-error {{not supported yet}}
    // FIXME: Need to check structure of lambda body 
    [](){ return 1; return 1; }; // expected-error {{not supported yet}}
  }
}
