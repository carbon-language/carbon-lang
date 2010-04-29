// RUN: %clang_cc1 -fsyntax-only -verify %s

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


N::M::template; // expected-error{{expected unqualified-id}}
N::M::template Promote; // expected-error{{expected unqualified-id}}

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

template<typename T>
struct TestA {
  typedef typename N::template B<T>::type type; // expected-error{{'B' following the 'template' keyword does not refer to a template}} \
                                                // expected-error{{expected member name}}
};

// Reduced from a Boost failure.
namespace test1 {
  template <class T> struct pair {
    T x;
    T y;

    static T pair<T>::* const mem_array[2];
  };

  template <class T>
  T pair<T>::* const pair<T>::mem_array[2] = { &pair<T>::x, &pair<T>::y };
}
