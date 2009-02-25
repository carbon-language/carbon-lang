// RUN: clang -fsyntax-only -verify %s

namespace N { 
  namespace M {
    template<typename T> struct Promote; // expected-note{{previous definition is here}}
    
    template<> struct Promote<short> {
      typedef int type;
    };
    
    template<> struct Promote<int> {
      typedef int type;
    };
    
    template<> struct Promote<float> {
      typedef double type;
    };
    
    Promote<short>::type *ret_intptr(int* ip) { return ip; }
    Promote<int>::type *ret_intptr2(int* ip) { return ip; }
  }

  M::Promote<int>::type *ret_intptr3(int* ip) { return ip; }
  M::template Promote<int>::type *ret_intptr4(int* ip) { return ip; }
}

N::M::Promote<int>::type *ret_intptr5(int* ip) { return ip; }
::N::M::Promote<int>::type *ret_intptr6(int* ip) { return ip; }


N::M::template; // expected-error{{expected template name after 'template' keyword in nested name specifier}} \
               // expected-error{{expected unqualified-id}}

N::M::template Promote; // expected-error{{expected '<' after 'template Promote' in nested name specifier}} \
// expected-error{{C++ requires a type specifier for all declarations}} \
// expected-error{{redefinition of 'Promote' as different kind of symbol}} \
// expected-error{{no member named 'Promote'}}

namespace N {
  template<typename T> struct A;

  template<>
  struct A<int> {
    struct X;
  };
}

struct ::N::A<int>::X {
  int foo;
};
