// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -fixit -o - | %clang_cc1 -fsyntax-only -pedantic -Werror -x c++ -
namespace std {
  template<typename T> class basic_string { 
    int find(const char *substr);
    static const int npos = -1;
  };

  typedef basic_string<char> string;
}

namespace otherstd {
  using namespace std;
}

using namespace std;

otherstd::strng str1; // expected-error{{no type named 'strng' in namespace 'otherstd'; did you mean 'string'?}}
tring str2; // expected-error{{unknown type name 'tring'; did you mean 'string'?}}

float area(float radius, float pi) {
  return radious * pi; // expected-error{{use of undeclared identifier 'radious'; did you mean 'radius'?}}
}

bool test_string(std::string s) {
  return s.find("hello") == std::string::pos; // expected-error{{no member named 'pos' in 'class std::basic_string<char>'; did you mean 'npos'?}}
}
