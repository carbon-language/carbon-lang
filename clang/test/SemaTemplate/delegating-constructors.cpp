// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify 

namespace PR10457 {

  class string
  {
    string(const char* str, unsigned);

  public:
    template <unsigned N>
    string(const char (&str)[N])
      : string(str) {} // expected-error{{constructor for 'string<6>' creates a delegation cycle}}
  };

  void f() {
    string s("hello");
  }
}
