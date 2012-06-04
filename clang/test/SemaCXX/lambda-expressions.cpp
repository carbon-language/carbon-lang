// RUN: %clang_cc1 -std=c++0x -Wno-unused-value -fsyntax-only -verify -fblocks %s

namespace std { class type_info; };

namespace ExplicitCapture {
  class C {
    int Member;

    static void Overload(int);
    void Overload();
    virtual C& Overload(float);

    void ImplicitThisCapture() {
      [](){(void)Member;}; // expected-error {{'this' cannot be implicitly captured in this context}}
      [&](){(void)Member;};

      [this](){(void)Member;};
      [this]{[this]{};};
      []{[this]{};};// expected-error {{'this' cannot be implicitly captured in this context}}
      []{Overload(3);};
      []{Overload();}; // expected-error {{'this' cannot be implicitly captured in this context}}
      []{(void)typeid(Overload());};
      []{(void)typeid(Overload(.5f));};// expected-error {{'this' cannot be implicitly captured in this context}}
    }
  };

  void f() {
    [this] () {}; // expected-error {{'this' cannot be captured in this context}}
  }
}

namespace ReturnDeduction {
  void test() {
    [](){ return 1; };
    [](){ return 1; };
    [](){ return ({return 1; 1;}); };
    [](){ return ({return 'c'; 1;}); }; // expected-error {{must match previous return type}}
    []()->int{ return 'c'; return 1; };
    [](){ return 'c'; return 1; };  // expected-error {{must match previous return type}}
    []() { return; return (void)0; };
    [](){ return 1; return 1; };
  }
}

namespace ImplicitCapture {
  void test() {
    int a = 0; // expected-note 5 {{declared}}
    []() { return a; }; // expected-error {{variable 'a' cannot be implicitly captured in a lambda with no capture-default specified}} expected-note {{begins here}}
    [&]() { return a; };
    [=]() { return a; };
    [=]() { int* b = &a; }; // expected-error {{cannot initialize a variable of type 'int *' with an rvalue of type 'const int *'}}
    [=]() { return [&]() { return a; }; };
    []() { return [&]() { return a; }; }; // expected-error {{variable 'a' cannot be implicitly captured in a lambda with no capture-default specified}} expected-note {{lambda expression begins here}}
    []() { return ^{ return a; }; };// expected-error {{variable 'a' cannot be implicitly captured in a lambda with no capture-default specified}} expected-note {{lambda expression begins here}}
    []() { return [&a] { return a; }; }; // expected-error 2 {{variable 'a' cannot be implicitly captured in a lambda with no capture-default specified}} expected-note 2 {{lambda expression begins here}}
    [=]() { return [&a] { return a; }; }; //

    const int b = 2;
    []() { return b; };

    union { // expected-note {{declared}}
      int c;
      float d;
    };
    d = 3;
    [=]() { return c; }; // expected-error {{unnamed variable cannot be implicitly captured in a lambda expression}}

    __block int e; // expected-note 3 {{declared}}
    [&]() { return e; }; // expected-error {{__block variable 'e' cannot be captured in a lambda expression}}
    [&e]() { return e; }; // expected-error 2 {{__block variable 'e' cannot be captured in a lambda expression}}

    int f[10]; // expected-note {{declared}}
    [&]() { return f[2]; };
    (void) ^{ return []() { return f[2]; }; }; // expected-error {{variable 'f' cannot be implicitly captured in a lambda with no capture-default specified}} \
    // expected-note{{lambda expression begins here}}

    struct G { G(); G(G&); int a; }; // expected-note 6 {{not viable}}
    G g;
    [=]() { const G* gg = &g; return gg->a; };
    [=]() { return [=]{ const G* gg = &g; return gg->a; }(); }; // expected-error {{no matching constructor for initialization of 'ImplicitCapture::G'}}
    (void)^{ return [=]{ const G* gg = &g; return gg->a; }(); }; // expected-error 2 {{no matching constructor for initialization of 'const ImplicitCapture::G'}}

    const int h = a; // expected-note {{declared}}
    []() { return h; }; // expected-error {{variable 'h' cannot be implicitly captured in a lambda with no capture-default specified}} expected-note {{lambda expression begins here}}

    // The exemption for variables which can appear in constant expressions
    // applies only to objects (and not to references).
    // FIXME: This might be a bug in the standard.
    static int i;
    constexpr int &ref_i = i; // expected-note {{declared}}
    [] { return ref_i; }; // expected-error {{variable 'ref_i' cannot be implicitly captured in a lambda with no capture-default specified}} expected-note {{lambda expression begins here}}
  }
}

namespace PR12031 {
  struct X {
    template<typename T>
    X(const T&);
    ~X();
  };

  void f(int i, X x);
  void g() {
    const int v = 10;
    f(v, [](){});
  }
}

namespace NullPtr {
  int &f(int *p);
  char &f(...);
  void g() {
    int n = 0;
    [=] {
      char &k = f(n); // not a null pointer constant
    } ();

    const int m = 0;
    [=] {
      int &k = f(m); // a null pointer constant
    } ();

    [=] () -> bool {
      int &k = f(m); // a null pointer constant
      return &m == 0;
    } ();

    [m] {
      int &k = f(m); // a null pointer constant
    } ();
  }
}

void PR12248()
{
  unsigned int result = 0;
  auto l = [&]() { ++result; };
}

namespace ModifyingCapture {
  void test() {
    int n = 0;
    [=] {
      n = 1; // expected-error {{cannot assign to a variable captured by copy in a non-mutable lambda}}
    };
  }
}
