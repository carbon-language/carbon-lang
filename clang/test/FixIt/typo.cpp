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

other_std::strng str1; // expected-error{{use of undeclared identifier 'other_std'; did you mean 'otherstd'?}} \
// expected-error{{no type named 'strng' in namespace 'otherstd'; did you mean 'string'?}}
tring str2; // expected-error{{unknown type name 'tring'; did you mean 'string'?}}

float area(float radius, float pi) {
  return radious * pi; // expected-error{{use of undeclared identifier 'radious'; did you mean 'radius'?}}
}

bool test_string(std::string s) {
  basc_string<char> b1; // expected-error{{no template named 'basc_string'; did you mean 'basic_string'?}}
  std::basic_sting<char> b2; // expected-error{{no template named 'basic_sting' in namespace 'std'; did you mean 'basic_string'?}}
  (void)b1;
  (void)b2;
  return s.fnd("hello") // expected-error{{no member named 'fnd' in 'class std::basic_string<char>'; did you mean 'find'?}}
    == std::string::pos; // expected-error{{no member named 'pos' in 'class std::basic_string<char>'; did you mean 'npos'?}}
}
