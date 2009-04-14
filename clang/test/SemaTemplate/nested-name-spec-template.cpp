// RUN: clang-cc -fsyntax-only -verify %s

namespace N { 
  namespace M {
    template<typename T> struct Promote;
    
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
// expected-error{{C++ requires a type specifier for all declarations}}

namespace N {
  template<typename T> struct A;

  template<>
  struct A<int> {
    struct X;
  };

  struct B;
}

struct ::N::A<int>::X {
  int foo;
};

#if 0
// FIXME: the following crashes the parser, because Sema has no way to
// communicate that the "dependent" template-name N::template B doesn't
// actually refer to a template.
template<typename T>
struct TestA {
  typedef typename N::template B<T>::type type; // xpected-error{{'B' following the 'template' keyword does not refer to a template}}
  // FIXME: should show what B *does* refer to.
};
#endif
