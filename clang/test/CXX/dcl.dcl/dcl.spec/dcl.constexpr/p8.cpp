// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct S {
  constexpr void f();
  constexpr void g() const;
};

void f(const S &s) {
  s.f();
  s.g();
}

namespace std_example {

  class debug_flag { // expected-note {{not an aggregate and has no constexpr constructors}}
  public:
    explicit debug_flag(bool);
    constexpr bool is_on(); // expected-error {{non-literal type 'std_example::debug_flag' cannot have constexpr members}}
  private:
    bool flag;
  };

  constexpr int bar(int x, int y) // expected-note {{here}}
    { return x + y + x*y; }
  int bar(int x, int y) // expected-error {{non-constexpr declaration of 'bar' follows constexpr declaration}}
    { return x * 2 + 3 * y; }

}
