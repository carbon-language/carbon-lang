// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -fixit -o - | %clang_cc1 -fsyntax-only -pedantic -Werror -x c++ -
namespace std {
  template<typename T> class basic_string { };
  typedef basic_string<char> string;
}

namespace otherstd {
  using namespace std;
}

using namespace std;

otherstd::strng str1; // expected-error{{no type named 'strng' in namespace 'otherstd'; did you mean 'string'?}}
tring str2; // expected-error{{unknown type name 'tring'; did you mean 'string'?}}
