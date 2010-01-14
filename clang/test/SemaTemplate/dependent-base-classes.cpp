// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T, typename U>
struct X0 : T::template apply<U> { 
  X0(U u) : T::template apply<U>(u) { }
};

template<typename T, typename U>
struct X1 : T::apply<U> { }; // expected-error{{missing 'template' keyword prior to dependent template name 'T::apply'}}

template<typename T>
struct X2 : vector<T> { }; // expected-error{{unknown template name 'vector'}}

namespace PR6031 {
  template<typename T>
  struct A;

  template <class X>
  struct C { };

  template <class TT>
  struct II {
    typedef typename A<TT>::type type;
  };

  template <class TT>
  struct FI : II<TT>
  {
    C<typename FI::type> a;
  };

  template <class TT>
  struct FI2
  {
    C<typename FI2::type> a; // expected-error{{no type named 'type' in 'struct PR6031::FI2'}} \
        // expected-error{{C++ requires a type specifier for all declarations}}
  };

  template<typename T>
  struct Base {
    class Nested { };
    template<typename U> struct MemberTemplate { };
    int a;
  };

  template<typename T>
  struct HasDepBase : Base<T> {
    int foo() {
      class HasDepBase::Nested nested;
      typedef typename HasDepBase::template MemberTemplate<T>::type type;
      return HasDepBase::a;
    }
  };

  template<typename T>
  struct NoDepBase {
    int foo() {
      class NoDepBase::Nested nested; // expected-error{{'Nested' does not name a tag member in the specified scope}}
      typedef typename NoDepBase::template MemberTemplate<T>::type type; // expected-error{{'MemberTemplate' following the 'template' keyword does not refer to a template}} \
      // FIXME: expected-error{{expected an identifier or template-id after '::'}} \
      // FIXME: expected-error{{unqualified-id}}
      return NoDepBase::a; // expected-error{{no member named 'a' in 'struct PR6031::NoDepBase'}}
    }
  };
}
