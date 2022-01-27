// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fexceptions -fcxx-exceptions -verify %s
// RUN: %clang_cc1 -std=c++2a -fsyntax-only -fexceptions -fcxx-exceptions -verify %s

template<typename... Types> struct tuple;
template<int I> struct int_c;

template<typename T>
struct identity {
  typedef T type;
};

template<typename T, typename U>
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};

// FIXME: Several more bullets to go

// In a function parameter pack, the pattern is the parameter-declaration
// without the ellipsis.
namespace PR11850 {
  template<typename ...T> struct S {
    int f(T...a, int b) { return b; }
  };
  S<> s;
  S<int*, char, const double&> t;
  int k = s.f(0);
  int l = t.f(&k, 'x', 5.9, 4);

  template<typename ...As> struct A {
    template<typename ...Bs> struct B {
      template<typename ...Cs> struct C {
        C(As..., Bs..., int &k, Cs...);
      };
    };
  };
  A<>::B<>::C<> c000(k);
  A<int>::B<>::C<int> c101(1, k, 3);
  A<>::B<int>::C<int> c011(1, k, 3);
  A<int>::B<int>::C<> c110(1, 2, k);
  A<int, int>::B<int, int>::C<int, int> c222(1, 2, 3, 4, k, 5, 6);
  A<int, int, int>::B<>::C<> c300(1, 2, 3, k);

  int &f();
  char &f(void*);
  template<typename ...A> struct U {
    template<typename ...B> struct V {
      auto g(A...a, B...b) -> decltype(f(a...));
    };
  };
  U<>::V<int*> v0;
  U<int*>::V<> v1;
  int &v0f = v0.g(0);
  char &v1f = v1.g(0);
}
namespace PR12096 {
  void Foo(int) {}
  void Foo(int, int) = delete;
  template<typename ...Args> struct Var {
    Var(const Args &...args, int *) { Foo(args...); }
  };
  Var<int> var(1, 0);
}

// In an initializer-list (8.5); the pattern is an initializer-clause.
// Note: this also covers expression-lists, since expression-list is
// just defined as initializer-list.
void five_args(int, int, int, int, int); // expected-note{{candidate function not viable: requires 5 arguments, but 6 were provided}}

template<int ...Values>
void initializer_list_expansion() {
  int values[5] = { Values... }; // expected-error{{excess elements in array initializer}}
  five_args(Values...); // expected-error{{no matching function for call to 'five_args'}}
}

template void initializer_list_expansion<1, 2, 3, 4, 5>();
template void initializer_list_expansion<1, 2, 3, 4, 5, 6>(); // expected-note{{in instantiation of function template specialization 'initializer_list_expansion<1, 2, 3, 4, 5, 6>' requested here}}

namespace PR8977 {
  struct A { };
  template<typename T, typename... Args> void f(Args... args) {
    // An empty expression-list performs value initialization.
    constexpr T t(args...);
  };

  template void f<A>();
}

// In a base-specifier-list (Clause 10); the pattern is a base-specifier.
template<typename ...Mixins>
struct HasMixins : public Mixins... { 
  HasMixins();
  HasMixins(const HasMixins&);
  HasMixins(int i);
};

struct A { }; // expected-note{{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'int' to 'const A' for 1st argument}} \
// expected-note{{candidate constructor (the implicit move constructor) not viable: no known conversion from 'int' to 'A' for 1st argument}} \
// expected-note{{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}
struct B { }; // expected-note{{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'int' to 'const B' for 1st argument}} \
// expected-note{{candidate constructor (the implicit move constructor) not viable: no known conversion from 'int' to 'B' for 1st argument}} \
// expected-note{{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}
struct C { };
struct D { };

A *checkA = new HasMixins<A, B, C, D>;
B *checkB = new HasMixins<A, B, C, D>;
D *checkD = new HasMixins<A, B, C, D>;
C *checkC = new HasMixins<A, B, D>; // expected-error{{cannot initialize a variable of type 'C *' with an rvalue of type 'HasMixins<A, B, D> *'}}
HasMixins<> *checkNone = new HasMixins<>;

template<typename Mixins>
struct BrokenMixins : public Mixins... { }; // expected-error{{pack expansion does not contain any unexpanded parameter packs}}

// In a mem-initializer-list (12.6.2); the pattern is a mem-initializer.
template<typename ...Mixins>
HasMixins<Mixins...>::HasMixins(): Mixins()... { }

template<typename ...Mixins>
HasMixins<Mixins...>::HasMixins(const HasMixins &other): Mixins(other)... { }

template<typename ...Mixins>
HasMixins<Mixins...>::HasMixins(int i): Mixins(i)... { }
// expected-error@-1 {{no matching constructor for initialization of 'A'}}
// expected-error@-2 {{no matching constructor for initialization of 'B'}}

void test_has_mixins() {
  HasMixins<A, B> ab;
  HasMixins<A, B> ab2 = ab;
  HasMixins<A, B> ab3(17); // expected-note{{in instantiation of member function 'HasMixins<A, B>::HasMixins' requested here}}
}

template<typename T>
struct X {
  T member;

  X() : member()... { } // expected-error{{pack expansion for initialization of member 'member'}}
};

// There was a bug in the delayed parsing code for the
// following case.
template<typename ...T>
struct DelayedParseTest : T...
{
  int a;
  DelayedParseTest(T... i) : T{i}..., a{10} {}
};


// In a template-argument-list (14.3); the pattern is a template-argument.
template<typename ...Types>
struct tuple_of_refs {
  typedef tuple<Types& ...> types;
};

tuple<int&, float&> *t_int_ref_float_ref;
tuple_of_refs<int&, float&>::types *t_int_ref_float_ref_2 =  t_int_ref_float_ref;

template<typename ...Types>
struct extract_nested_types {
  typedef tuple<typename Types::type...> types;
};

tuple<int, float> *t_int_float;
extract_nested_types<identity<int>, identity<float> >::types *t_int_float_2 
  = t_int_float;

template<int ...N>
struct tuple_of_ints {
  typedef tuple<int_c<N>...> type;
};

int check_temp_arg_1[is_same<tuple_of_ints<1, 2, 3, 4, 5>::type, 
                             tuple<int_c<1>, int_c<2>, int_c<3>, int_c<4>, 
                                   int_c<5>>>::value? 1 : -1];

#if __cplusplus < 201703L
// In a dynamic-exception-specification (15.4); the pattern is a type-id.
template<typename ...Types>
struct f_with_except {
  virtual void f() throw(Types...); // expected-note{{overridden virtual function is here}}
};

struct check_f_with_except_1 : f_with_except<int, float> {
  virtual void f() throw(int, float);
};

struct check_f_with_except_2 : f_with_except<int, float> {
  virtual void f() throw(int);
};

struct check_f_with_except_3 : f_with_except<int, float> {
  virtual void f() throw(int, float, double); // expected-error{{exception specification of overriding function is more lax than base version}}
};
#endif

namespace PackExpansionWithinLambda {
  void swallow(...);
  template<typename ...T, typename ...U> void f(U ...u) {
    swallow([=] {
      // C++17 [temp.variadic]p4:
      //   Pack expansions can occur in the following contexts:

      //    - in a function parameter pack
      void g(T...);

#if __cplusplus >= 201703L
      struct A : T... {
        //  - in a using-declaration
        using T::x...;
        using typename T::U...;
      };
#endif

#if __cplusplus > 201703L
      //    - in a template parameter pack that is a pack expansion
      swallow([]<T *...v, template<T *> typename ...W>(W<v> ...wv) { });
#endif

      //    - in an initializer-list
      int arr[] = {T().x...};

      //    - in a base-specifier-list
      struct B : T... {
        //  - in a mem-initializer-list
        B() : T{0}... {}
      };

      //    - in a template-argument-list
      f<T...>();

      //    - in an attribute-list
      // FIXME: We do not support any such attributes yet.
      
      //    - in an alignment-specifier
      alignas(T...) int y;

      //    - in a capture-list
      [](T ...t) { [t...]{}(); } (T()...);

      //    - in a sizeof... expression
      const int k1 = sizeof...(T);

#if __cplusplus >= 201703L
      //    - in a fold-expression
      const int k2 = ((sizeof(T)/sizeof(T)) + ...);

      static_assert(k1 == k2);
#endif

      // Trigger clang to look in here for unexpanded packs.
      U u;
    } ...);
  }

  template<typename ...T> void nested() {
    swallow([=] {
      [](T ...t) { [t]{}(); } (T()...); // expected-error {{unexpanded parameter pack 't'}}
    }...); // expected-error {{does not contain any unexpanded}}
  }

  template <typename ...T> void g() {
    // Check that we do detect the above cases when the pack is not expanded.
    swallow([=] { void h(T); }); // expected-error {{unexpanded parameter pack 'T'}}
    swallow([=] { struct A : T {}; }); // expected-error {{unexpanded parameter pack 'T'}}
#if __cplusplus >= 201703L
    swallow([=] { struct A : T... { using T::x; }; }); // expected-error {{unexpanded parameter pack 'T'}}
    swallow([=] { struct A : T... { using typename T::U; }; }); // expected-error {{unexpanded parameter pack 'T'}}
#endif

    swallow([=] { int arr[] = {T().x}; }); // expected-error {{unexpanded parameter pack 'T'}}
    swallow([=] { struct B : T... { B() : T{0} {} }; }); // expected-error {{unexpanded parameter pack 'T'}}
    swallow([=] { f<T>(); }); // expected-error {{unexpanded parameter pack 'T'}}
    swallow([=] { alignas(T) int y; }); // expected-error {{unexpanded parameter pack 'T'}}
    swallow([=] { [](T ...t) {
          [t]{}(); // expected-error {{unexpanded parameter pack 't'}}
        } (T()...); });
  }

  struct T { int x; using U = int; };
  void g() { f<T>(1, 2, 3); }

  template<typename ...T> void pack_expand_attr() {
    // FIXME: Move this test into 'f' above once we support this.
    [[gnu::aligned(alignof(T))...]] int x; // expected-error {{cannot be used as an attribute pack}} expected-error {{unexpanded}}
  }
}
