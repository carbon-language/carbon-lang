// RUN: %clang_cc1 -verify -std=c++11 %s

namespace N {
  typedef char C;
}

namespace M {
  typedef double D;
}

struct NonLiteral {
  NonLiteral() {}
  NonLiteral(int) {}
  operator int() const { return 0; }
};
struct Literal {
  constexpr Literal() {}
  operator int() const { return 0; }
};

struct S {
  virtual int ImplicitlyVirtual();
};
struct T {};

template<typename T> struct ImplicitVirtualFromDependentBase : T {
  constexpr int ImplicitlyVirtual() { return 0; }
};

// FIXME: Can't test this until we have function invocation substitution
#if 0
constexpr int a = ImplicitVirtualFromDependentBase<S>().ImplicitlyVirtual(); // desired-error {{not a constant expression}}
constexpr int b = ImplicitVirtualFromDependentBase<T>().ImplicitlyVirtual(); // ok
#endif

template<typename R> struct ConstexprMember {
  constexpr R F() { return 0; }
};
// FIXME: Can't test this until we have function invocation substitution
#if 0
constexpr int c = ConstexprMember<int>().F(); // ok
constexpr int d = ConstexprMember<NonLiteral>().F(); // desired-error {{not a constant expression}}
#endif

template<typename ...P> struct ConstexprCtor { // expected-note 2{{no constexpr constructors}}
  constexpr ConstexprCtor(P...); // expected-note {{constructor template instantiation is not constexpr because 1st parameter type 'NonLiteral' is not a literal type}} \
                                    expected-note {{constructor template instantiation is not constexpr because 2nd parameter type 'NonLiteral' is not a literal type}}
};
constexpr ConstexprCtor<> f1(); // ok
constexpr ConstexprCtor<int> f2(); // ok
constexpr ConstexprCtor<NonLiteral> f3(); // expected-error {{not a literal type}}
constexpr ConstexprCtor<int, NonLiteral> f4(); // expected-error {{not a literal type}}

struct VirtBase : virtual S {}; // expected-note {{here}}

namespace TemplateVBase {
  template<typename T> struct T1 : virtual Literal { // expected-note {{here}}
    constexpr T1(); // expected-error {{constexpr constructor not allowed in struct with virtual base class}}
  };

  template<typename T> struct T2 : virtual T { // expected-note {{struct with virtual base class is not a literal type}} expected-note {{here}}
    // FIXME: This is ill-formed (no diagnostic required).
    // We should diagnose it now rather than waiting until instantiation.
    constexpr T2(); // desired-error {{constexpr constructor not allowed in class with virtual base classes}}
  };
  constexpr T2<Literal> g2(); // expected-error {{not a literal type}}

  template<typename T> class T3 : public T { // expected-note {{class with virtual base class is not a literal type}}
  public:
    constexpr T3() {}
  };
  constexpr T3<Literal> g3(); // ok
  constexpr T3<VirtBase> g4(); // expected-error {{not a literal type}}
}
