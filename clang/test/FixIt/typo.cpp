// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -fsyntax-only -fixit -x c++ %t
// RUN: %clang_cc1 -fsyntax-only -pedantic -Werror -x c++ %t
// RUN: grep test_string %t

namespace std {
  template<typename T> class basic_string { // expected-note 2{{'basic_string' declared here}}
  public:
    int find(const char *substr); // expected-note{{'find' declared here}}
    static const int npos = -1; // expected-note{{'npos' declared here}}
  };

  typedef basic_string<char> string; // expected-note 2{{'string' declared here}}
}

namespace otherstd { // expected-note 2{{'otherstd' declared here}} \
                     // expected-note{{namespace 'otherstd' defined here}}
  using namespace std;
}

using namespace std;

other_std::strng str1; // expected-error{{use of undeclared identifier 'other_std'; did you mean 'otherstd'?}} \
// expected-error{{no type named 'strng' in namespace 'otherstd'; did you mean 'string'?}}
tring str2; // expected-error{{unknown type name 'tring'; did you mean 'string'?}}

::other_std::string str3; // expected-error{{no member named 'other_std' in the global namespace; did you mean 'otherstd'?}}

float area(float radius, // expected-note{{'radius' declared here}}
           float pi) {
  return radious * pi; // expected-error{{did you mean 'radius'?}}
}

using namespace othestd; // expected-error{{no namespace named 'othestd'; did you mean 'otherstd'?}}
namespace blargh = otherstd; // expected-note 3{{namespace 'blargh' defined here}}
using namespace ::blarg; // expected-error{{no namespace named 'blarg' in the global namespace; did you mean 'blargh'?}}

namespace wibble = blarg; // expected-error{{no namespace named 'blarg'; did you mean 'blargh'?}}
namespace wobble = ::blarg; // expected-error{{no namespace named 'blarg' in the global namespace; did you mean 'blargh'?}}

bool test_string(std::string s) {
  basc_string<char> b1; // expected-error{{no template named 'basc_string'; did you mean 'basic_string'?}}
  std::basic_sting<char> b2; // expected-error{{no template named 'basic_sting' in namespace 'std'; did you mean 'basic_string'?}}
  (void)b1;
  (void)b2;
  return s.fnd("hello") // expected-error{{no member named 'fnd' in 'std::basic_string<char>'; did you mean 'find'?}}
    == std::string::pos; // expected-error{{no member named 'pos' in 'std::basic_string<char>'; did you mean 'npos'?}}
}

struct Base { };
struct Derived : public Base { // expected-note{{base class 'Base' specified here}}
  int member; // expected-note 3{{'member' declared here}}

  Derived() : base(), // expected-error{{initializer 'base' does not name a non-static data member or base class; did you mean the base class 'Base'?}}
              ember() { } // expected-error{{initializer 'ember' does not name a non-static data member or base class; did you mean the member 'member'?}}

  int getMember() const {
    return ember; // expected-error{{use of undeclared identifier 'ember'; did you mean 'member'?}}
  }

  int &getMember();
};

int &Derived::getMember() {
  return ember; // expected-error{{use of undeclared identifier 'ember'; did you mean 'member'?}}
}
