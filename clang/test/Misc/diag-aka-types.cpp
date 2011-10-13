// RUN: %clang_cc1 %s -fsyntax-only -verify -std=c++11

struct X {};
typedef X foo_t;

foo_t *ptr;
char c1 = ptr; // expected-error{{'foo_t *' (aka 'X *')}}

const foo_t &ref = foo_t();
char c2 = ref; // expected-error{{'const foo_t' (aka 'const X')}}

// deduced auto should not produce an aka.
auto aut = X();
char c3 = aut; // expected-error{{from 'X' to 'char'}}

// There are two classes named Foo::foo here.  Make sure the message gives
// a way to them apart.
namespace Foo {
  class foo {};
}

namespace bar {
  namespace Foo {
    class foo;
  }
  void f(Foo::foo* x);  // expected-note{{passing argument to parameter 'x' here}}
}

void test(Foo::foo* x) {
  bar::f(x); // expected-error{{cannot initialize a parameter of type 'Foo::foo *' (aka 'bar::Foo::foo *') with an lvalue of type 'Foo::foo *')}}
}

// PR9548 - "no known conversion from 'vector<string>' to 'vector<string>'"
// vector<string> refers to two different types here.  Make sure the message
// gives a way to tell them apart.
class versa_string;
typedef versa_string string;

namespace std {template <typename T> class vector;}
using std::vector;

void f(vector<string> v);  // expected-note {{candidate function not viable: no known conversion from 'vector<string>' (aka 'std::vector<std::basic_string>') to 'vector<string>' (aka 'std::vector<versa_string>') for 1st argument}}

namespace std {
  class basic_string;
  typedef basic_string string;
  template <typename T> class vector {};
  void g() {
    vector<string> v;
    f(v);  // expected-error{{no matching function for call to 'f'}}
  }
}
