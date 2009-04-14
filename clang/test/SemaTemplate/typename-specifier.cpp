// RUN: clang-cc -fsyntax-only -verify %s
namespace N {
  struct A {
    typedef int type;
  };

  struct B {
  };

  struct C {
    struct type { };
    int type; // expected-note 2{{referenced member 'type' is declared here}}
  };
}

int i;

typename N::A::type *ip1 = &i;
typename N::B::type *ip2 = &i; // expected-error{{ no type named 'type' in 'B'}}
typename N::C::type *ip3 = &i; // expected-error{{typename specifier refers to non-type member 'type'}}

void test(double d) {
  typename N::A::type f(typename N::A::type(a)); // expected-warning{{parentheses were disambiguated as a function declarator}}
  int five = f(5);
  
  using namespace N;
  for (typename A::type i = 0; i < 10; ++i)
    five += 1;

  const typename N::A::type f2(d);
}

namespace N {
  template<typename T>
  struct X {
    typedef typename T::type type; // expected-error 2{{no type named 'type' in 'B'}} \
    // FIXME: location info for error above isn't very good \
    // expected-error 2{{typename specifier refers to non-type member 'type'}} \
    // expected-error{{type 'int' cannot be used prior to '::' because it has no members}}
  };
}

N::X<N::A>::type *ip4 = &i;
N::X<N::B>::type *ip5 = &i; // expected-note{{in instantiation of template class 'struct N::X<struct N::B>' requested here}} \
// expected-error{{unknown type name 'type'}}
N::X<N::C>::type *ip6 = &i; // expected-note{{in instantiation of template class 'struct N::X<struct N::C>' requested here}} \
// expected-error{{unknown type name 'type'}}

N::X<int>::type fail1; // expected-note{{in instantiation of template class 'struct N::X<int>' requested here}} \
// expected-error{{unknown type name 'type'}}

template<typename T>
struct Y {
  typedef typename N::X<T>::type *type; // expected-note{{in instantiation of template class 'struct N::X<struct B>' requested here}} \
  // expected-note{{in instantiation of template class 'struct N::X<struct C>' requested here}}
};

struct A {
  typedef int type;
};

struct B {
};

struct C {
  struct type { };
  int type; // expected-note{{referenced member 'type' is declared here}}
};

::Y<A>::type ip7 = &i;
::Y<B>::type ip8 = &i; // expected-note{{in instantiation of template class 'struct Y<struct B>' requested here}} \
// expected-error{{unknown type name 'type'}}
::Y<C>::type ip9 = &i; // expected-note{{in instantiation of template class 'struct Y<struct C>' requested here}} \
// expected-error{{unknown type name 'type'}}
