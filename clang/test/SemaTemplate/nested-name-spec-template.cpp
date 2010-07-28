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
  M::template Promote<int>::type *ret_intptr4(int* ip) { return ip; } // expected-warning{{'template' keyword outside of a template}}
  M::template Promote<int> pi; // expected-warning{{'template' keyword outside of a template}}
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

typedef int T;
namespace N1 {
  template<typename T> T f0();
}

template<typename T> T N1::f0() { }

namespace PR7385 {
  template< typename > struct has_xxx0
  {
    template< typename > struct has_xxx0_introspect
    {
      template< typename > struct has_xxx0_substitute ;
      template< typename V > 
      int int00( has_xxx0_substitute < typename V::template xxx< > > = 0 );
    };
    static const int value = has_xxx0_introspect<int>::value; // expected-error{{no member named 'value'}}
    typedef int type;
  };

  has_xxx0<int>::type t; // expected-note{{instantiation of}}
}

namespace PR7725 {
  template<class ignored> struct TypedefProvider;
  template<typename T>
  struct TemplateClass : public TypedefProvider<T>
  {
    void PrintSelf() {
      TemplateClass::Test::PrintSelf();
    }
  };
}
