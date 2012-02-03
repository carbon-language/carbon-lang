// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify -fblocks %s

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

namespace ImplicitCapture {
  void test() {
    int a = 0; // expected-note 3 {{declared}}
    []() { return a; }; // expected-error {{variable 'a' cannot be implicitly captured in a lambda with no capture-default specified}} expected-note {{begins here}} expected-error {{not supported yet}} 
    [&]() { return a; }; // expected-error {{not supported yet}}
    [=]() { return a; }; // expected-error {{not supported yet}}
    [=]() { int* b = &a; }; // expected-error {{cannot initialize a variable of type 'int *' with an rvalue of type 'const int *'}} expected-error {{not supported yet}}
    [=]() { return [&]() { return a; }; }; // expected-error 2 {{not supported yet}}
    []() { return [&]() { return a; }; }; // expected-error {{variable 'a' cannot be implicitly captured in a lambda with no capture-default specified}} expected-note {{lambda expression begins here}} expected-error 2 {{not supported yet}}
    []() { return ^{ return a; }; };// expected-error {{variable 'a' cannot be implicitly captured in a lambda with no capture-default specified}} expected-note {{lambda expression begins here}} expected-error {{not supported yet}}

    const int b = 2;
    []() { return b; }; // expected-error {{not supported yet}}

    union { // expected-note {{declared}}
      int c;
      float d;
    };
    d = 3;
    [=]() { return c; }; // expected-error {{unnamed variable cannot be implicitly captured in a lambda expression}} expected-error {{not supported yet}}

    __block int e; // expected-note {{declared}}
    [&]() { return e; }; // expected-error {{__block variable 'e' cannot be captured in a lambda expression}} expected-error {{not supported yet}}

    int f[10]; // expected-note {{declared}}
    [&]() { return f[2]; };  // expected-error {{not supported yet}}
    (void) ^{ return []() { return f[2]; }; }; // expected-error {{cannot refer to declaration with an array type inside block}} expected-error {{not supported yet}}

    struct G { G(); G(G&); int a; }; // expected-note 6 {{not viable}}
    G g;
    [=]() { const G* gg = &g; return gg->a; }; // expected-error {{not supported yet}}
    [=]() { return [=]{ const G* gg = &g; return gg->a; }(); }; // expected-error {{no matching constructor for initialization of 'const ImplicitCapture::G'}} expected-error 2 {{not supported yet}}
    (void)^{ return [=]{ const G* gg = &g; return gg->a; }(); }; // expected-error 2 {{no matching constructor for initialization of 'const ImplicitCapture::G'}} expected-error {{not supported yet}}
  }
}
