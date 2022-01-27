// RUN: %clang_cc1 %s -I%S -std=c++2a -verify

namespace std { struct type_info; }

static_assert(requires { 0; });
static_assert(requires { "aaaa"; });
static_assert(requires { (0).da; }); // expected-error{{member reference base type 'int' is not a structure or union}}

struct A {};
struct B {
    B operator+(const B &other) const { return other; }
};
struct C {
    C operator+(C &other) const { return other; }
};

template<typename T> requires requires (T a, const T& b) { a + b; }
// expected-note@-1{{because 'a + b' would be invalid: invalid operands to binary expression ('A' and 'const A')}}
// expected-note@-2{{because 'a + b' would be invalid: invalid operands to binary expression ('C' and 'const C')}}
struct r1 {};

using r1i1 = r1<int>;
using r1i2 = r1<A>; // expected-error{{constraints not satisfied for class template 'r1' [with T = A]}}
using r1i3 = r1<B>;
using r1i4 = r1<C>; // expected-error{{constraints not satisfied for class template 'r1' [with T = C]}}

struct D { void foo() {} };

template<typename T> requires requires (T a) { a.foo(); }
// expected-note@-1{{because 'a.foo()' would be invalid: no member named 'foo' in 'A'}}
// expected-note@-2{{because 'a.foo()' would be invalid: member reference base type 'int' is not a structure or union}}
// expected-note@-3{{because 'a.foo()' would be invalid: 'this' argument to member function 'foo' has type 'const D', but function is not marked const}}
struct r2 {};

using r2i1 = r2<int>; // expected-error{{constraints not satisfied for class template 'r2' [with T = int]}}
using r2i2 = r2<A>; // expected-error{{constraints not satisfied for class template 'r2' [with T = A]}}
using r2i3 = r2<D>;
using r2i4 = r2<const D>; // expected-error{{constraints not satisfied for class template 'r2' [with T = const D]}}

template<typename T> requires requires { sizeof(T); }
// expected-note@-1{{because 'sizeof(T)' would be invalid: invalid application of 'sizeof' to an incomplete type 'void'}}
// expected-note@-2{{because 'sizeof(T)' would be invalid: invalid application of 'sizeof' to an incomplete type 'nonexistent'}}
struct r3 {};

using r3i1 = r3<int>;
using r3i2 = r3<A>;
using r3i3 = r3<A &>;
using r3i4 = r3<void>; // expected-error{{constraints not satisfied for class template 'r3' [with T = void]}}
using r3i4 = r3<class nonexistent>; // expected-error{{constraints not satisfied for class template 'r3' [with T = nonexistent]}}

template<typename T> requires requires (T t) { 0; "a"; (void)'a'; }
struct r4 {};

using r4i1 = r4<int>;
using r4i2 = r4<int[10]>;
using r4i3 = r4<int(int)>;

template<class T> void f(T) = delete;
template<class T> requires (sizeof(T) == 1) void f(T) { }

template<typename T> requires requires(T t) { f(t); }
// expected-note@-1{{because 'f(t)' would be invalid: call to deleted function 'f'}}
struct r5 {};

using r5i1 = r5<int>;
// expected-error@-1 {{constraints not satisfied for class template 'r5' [with T = int]}}
using r5i2 = r5<char>;

template<typename T>
struct E {
  struct non_default_constructible { non_default_constructible(T t) { } };
};

template<typename T> requires requires(T t) { typename E<T>::non_default_constructible{}; }
// expected-note@-1 {{because 'typename E<T>::non_default_constructible({})' would be invalid: no matching constructor for initialization of 'typename E<int>::non_default_constructible'}}
struct r6 {};

using r6i1 = r6<int>;
// expected-error@-1 {{constraints not satisfied for class template 'r6' [with T = int]}}

template<typename T> requires requires(T t) { typename E<T>::non_default_constructible(); }
// expected-note@-1 {{because 'typename E<T>::non_default_constructible()' would be invalid: no matching constructor for initialization of 'typename E<int>::non_default_constructible'}}
struct r7 {};

using r7i1 = r7<int>;
// expected-error@-1 {{constraints not satisfied for class template 'r7' [with T = int]}}

// C++ [expr.prim.req.simple] Example
namespace std_example {
  template<typename T> concept C =
    requires (T a, T b) { // expected-note{{because substituted constraint expression is ill-formed: argument may not have 'void' type}}
      a + b; // expected-note{{because 'a + b' would be invalid: invalid operands to binary expression ('int *' and 'int *')}}
    };

  static_assert(C<int>);
  template<C T> struct C_check {}; // expected-note{{because 'void' does not satisfy 'C'}} expected-note{{because 'int *' does not satisfy 'C'}}
  using c1c1 = C_check<void>; // expected-error{{constraints not satisfied for class template 'C_check' [with T = void]}}
  using c1c2 = C_check<int *>; // expected-error{{constraints not satisfied for class template 'C_check' [with T = int *]}}
}

// typeid() of an expression becomes potentially evaluated if the expression is
// of a polymorphic type.
class X { virtual ~X(); };
constexpr bool b = requires (X &x) { static_cast<int(*)[(typeid(x), 0)]>(nullptr); };
// expected-error@-1{{constraint variable 'x' cannot be used in an evaluated context}}
// expected-note@-2{{'x' declared here}}
