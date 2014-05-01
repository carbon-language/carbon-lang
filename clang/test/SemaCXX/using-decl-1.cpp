// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

extern "C" { void f(bool); }

namespace std {
  using ::f;
  inline void f() { return f(true); }
}

namespace M {
  void f(float);
}

namespace N {
  using M::f;
  void f(int) { } // expected-note{{previous}}
  
  void f(int) { } // expected-error{{redefinition}}
}

namespace N {
  void f(double);
  void f(long);
}

struct X0 {
  void operator()(int);
  void operator()(long);
};

struct X1 : X0 {
  // FIXME: give this operator() a 'float' parameter to test overloading
  // behavior. It currently fails.
  void operator()();
  using X0::operator();
  
  void test() {
    (*this)(1);
  }
};

struct A { void f(); };
struct B : A { };
class C : B { using B::f; };

// PR5751: Resolve overloaded functions through using decls.
namespace O {
  void f(int i);
  void f(double d);
}
namespace P {
  void f();
  void g(void (*ptr)(int));
  using O::f;
  void test() {
    f();
    f(1);
    void (*f_ptr1)(double) = f;
    void (*f_ptr2)() = f;
    g(f);
  }
}

// Make sure that ADL can find names brought in by using decls.
namespace test0 {
  namespace ns {
    class Foo {};
    
    namespace inner {
      void foo(char *); // expected-note {{no known conversion}} 
    }

    using inner::foo;
  }

  void test(ns::Foo *p) {
    foo(*p); // expected-error {{no matching function for call to 'foo'}}
  }
}

// Redeclarations!
namespace test1 {
  namespace ns0 { struct Foo {}; }
  namespace A { void foo(ns0::Foo *p, int y, int z); }
  namespace ns2 { using A::foo; }
  namespace ns1 { struct Bar : ns0::Foo {}; }
  namespace A { void foo(ns0::Foo *p, int y, int z = 0); } // expected-note {{candidate}}
  namespace ns1 { using A::foo; }
  namespace ns2 { struct Baz : ns1::Bar {}; }
  namespace A { void foo(ns0::Foo *p, int y = 0, int z); }

  void test(ns2::Baz *p) {
    foo(p, 0, 0); // okay!
    foo(p, 0); // should be fine!
    foo(p); // expected-error {{no matching function}}
  }
}

namespace test2 {
  namespace ns { int foo; }
  template <class T> using ns::foo; // expected-error {{cannot template a using declaration}}

  // PR8022
  struct A {
    template <typename T> void f(T);
  };
  class B : A {
    template <typename T> using A::f<T>; // expected-error {{cannot template a using declaration}}
  };
}

// PR8756
namespace foo
{
  class Class1; // expected-note{{forward declaration}}
  class Class2
  {
    using ::foo::Class1::Function; // expected-error{{incomplete type 'foo::Class1' named in nested name specifier}}
  };
}

// Don't suggest non-typenames for positions requiring typenames.
namespace using_suggestion_tyname_val {
namespace N { void FFF() {} }
using typename N::FFG; // expected-error {{no member named 'FFG' in namespace 'using_suggestion_tyname_val::N'}}
}

namespace using_suggestion_member_tyname_val {
class CCC { public: void AAA() { } };
class DDD : public CCC { public: using typename CCC::AAB; }; // expected-error {{no member named 'AAB' in 'using_suggestion_member_tyname_val::CCC'}}
}

namespace using_suggestion_tyname_val_dropped_specifier {
void FFF() {}
namespace N { }
using typename N::FFG; // expected-error {{no member named 'FFG' in namespace 'using_suggestion_tyname_val_dropped_specifier::N'}}
}

// Currently hints aren't provided to drop out the incorrect M::.
namespace using_suggestion_ty_dropped_nested_specifier {
namespace N {
class AAA {}; // expected-note {{'N::AAA' declared here}}
namespace M { }
}
using N::M::AAA; // expected-error {{no member named 'AAA' in namespace 'using_suggestion_ty_dropped_nested_specifier::N::M'; did you mean 'N::AAA'?}}
}

namespace using_suggestion_tyname_ty_dropped_nested_specifier {
namespace N {
class AAA {}; // expected-note {{'N::AAA' declared here}}
namespace M { }
}
using typename N::M::AAA; // expected-error {{no member named 'AAA' in namespace 'using_suggestion_tyname_ty_dropped_nested_specifier::N::M'; did you mean 'N::AAA'?}}
}

namespace using_suggestion_val_dropped_nested_specifier {
namespace N {
void FFF() {} // expected-note {{'N::FFF' declared here}}
namespace M { }
}
using N::M::FFF; // expected-error {{no member named 'FFF' in namespace 'using_suggestion_val_dropped_nested_specifier::N::M'; did you mean 'N::FFF'?}}
}

namespace UsingDeclVsHiddenName {
  namespace A {
    enum HiddenTag1 {}; // expected-note {{previous use is here}}
    enum HiddenTag2 {}; // expected-note {{target}}
    int HiddenFn1; // expected-note {{target}}
    int HiddenFn2; // expected-note {{target}}
    int HiddenLocalExtern1;
    int HiddenLocalExtern2;
  }

  namespace B {
    using A::HiddenTag1;
    using A::HiddenFn1; // expected-note {{using declaration}}
    using A::HiddenLocalExtern1;

    struct S {
      friend struct HiddenTag1; // expected-error {{tag type that does not match previous}}
      friend struct HiddenTag2; // expected-note {{conflicting declaration}}
      friend void HiddenFn1(); // expected-error {{cannot befriend target of using declaration}}
      friend void HiddenFn2(); // expected-note {{conflicting declaration}}
      void f() {
        // OK, these are not in the scope of namespace B, even though they're
        // members of the namespace.
        void HiddenLocalExtern1();
        void HiddenLocalExtern2();
      }
    };

    using A::HiddenTag2; // expected-error {{conflicts with declaration already in scope}}
    using A::HiddenFn2; // expected-error {{conflicts with declaration already in scope}}
    using A::HiddenLocalExtern2;
  }
}

namespace PR19171 {
  struct Z {
    Z();
  };

  typedef struct {
    Z i;
  } S;

  struct Y : S {
    using S::S;
#if __cplusplus < 201103L
    // expected-error@-2 {{no member named 'S' in 'PR19171::S'}}
#endif
  };

  // [namespace.udecl]p3: In a using-declaration used as a member-declaration,
  // the nested-name-specifier shall name a base class of the class being defined.
  // If such a using-declaration names a constructor, the nested-name-specifier
  // shall name a direct base class of the class being defined;

  struct B_blah { };
  struct C_blah : B_blah { C_blah(int); }; // expected-note 0-1{{declared here}}
  struct D1 : C_blah {
    // FIXME: We should be able to correct this in C++11 mode.
    using B_blah::C_blah; // expected-error-re {{no member named 'C_blah' in 'PR19171::B_blah'{{$}}}}
  };
  struct D2 : C_blah {
    // Somewhat bizarrely, this names the injected-class-name of B_blah within
    // C_blah, and is valid.
    using C_blah::B_blah;
  };
  struct D3 : C_blah {
    using C_blah::D_blah;
#if __cplusplus < 201103L
    // expected-error-re@-2 {{no member named 'D_blah' in 'PR19171::C_blah'{{$}}}}
#else
    // expected-error@-4 {{no member named 'D_blah' in 'PR19171::C_blah'; did you mean 'C_blah'?}}
#endif
  };
#if __cplusplus >= 201103L
  D3 d3(0); // ok
#endif

  struct E { };
  struct EE { int EE; };
  struct F : E {
    using E::EE; // expected-error-re {{no member named 'EE' in 'PR19171::E'{{$}}}}
  };
}

namespace TypoCorrectTemplateMember {
  struct A {
    template<typename T> void foobar(T); // expected-note {{'foobar' declared here}}
  };
  struct B : A {
    using A::goobar; // expected-error {{no member named 'goobar' in 'TypoCorrectTemplateMember::A'; did you mean 'foobar'?}}
  };
}
