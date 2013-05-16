// RUN: %clang_cc1 -std=c++1y %s -verify

// For every init-capture a non-static data member named by the identifier of
// the init-capture is declared in the closure type.
const char *has_member_x = [x("hello")] {}.x;
// This member is not a bit-field...
auto capturing_lambda = [n(0)] {};
int decltype(capturing_lambda)::*mem_ptr = &decltype(capturing_lambda)::n;
// ... and not mutable.
const auto capturing_lambda_copy = capturing_lambda;
int &n = capturing_lambda_copy.n; // expected-error {{drops qualifiers}}

// The type of that member [...is that of a...] variable declaration of the form
// "auto init-capture ;"...
auto with_float = [f(1.0f)] {};
float &f = with_float.f;
// ... except that the variable name is replaced by a unique identifier.
auto with_float_2 = [&f(f)] {}; // ok, refers to outer f
float &f2 = with_float_2.f;

// Within the lambda-expression's lambda-declarator (FIXME) and
// compound-statement, the identifier in the init-capture hides any declaration
// of the same name in scopes enclosing the lambda-expression.
void hiding() {
  char c;
  (void) [c("foo")] {
    static_assert(sizeof(c) == sizeof(const char*), "");
  };
  (void) [c("bar")] () -> decltype(c) {
    // FIXME: the 'c' in the return type should be the init-capture, not the
    // outer c.
    return "baz"; // expected-error {{cannot initialize}}
  };
}

struct ExplicitCopy {
  ExplicitCopy(); // expected-note 2{{not viable}}
  explicit ExplicitCopy(const ExplicitCopy&);
};
auto init_kind_1 = [ec(ExplicitCopy())] {};
auto init_kind_2 = [ec = ExplicitCopy()] {}; // expected-error {{no matching constructor}}

template<typename T> void init_kind_template() {
  auto init_kind_1 = [ec(T())] {};
  auto init_kind_2 = [ec = T()] {}; // expected-error {{no matching constructor}}
}
template void init_kind_template<int>();
template void init_kind_template<ExplicitCopy>(); // expected-note {{instantiation of}}

void void_fn();
int overload_fn();
int overload_fn(int);

auto bad_init_1 = [a()] {}; // expected-error {{expected expression}}
auto bad_init_2 = [a(1, 2)] {}; // expected-error {{initializer for lambda capture 'a' contains multiple expressions}}
auto bad_init_3 = [&a(void_fn())] {}; // expected-error {{cannot form a reference to 'void'}}
auto bad_init_4 = [a(void_fn())] {}; // expected-error {{field has incomplete type 'void'}}
auto bad_init_5 = [a(overload_fn)] {}; // expected-error {{cannot deduce type for lambda capture 'a' from initializer of type '<overloaded function}}
auto bad_init_6 = [a{overload_fn}] {}; // expected-error {{cannot deduce type for lambda capture 'a' from initializer list}}

template<typename...T> void pack_1(T...t) { [a(t...)] {}; } // expected-error {{initializer missing for lambda capture 'a'}}
template void pack_1<>(); // expected-note {{instantiation of}}

auto multi_return(int a, int b) {
  return [n(a + 2*b), m(a - 2*b)] {};
}
auto use_multi_return() {
  auto nm = multi_return(5, 9);
  return nm.n + nm.m;
}

auto a = [a(4), b = 5, &c = static_cast<const int&&>(0)] {
  static_assert(sizeof(a) == sizeof(int), "");
  static_assert(sizeof(b) == sizeof(int), "");
  using T = decltype(c);
  using T = const int &;
};
auto b = [a{0}] {}; // expected-error {{include <initializer_list>}}

struct S { S(); S(S&&); };
template<typename T> struct remove_reference { typedef T type; };
template<typename T> struct remove_reference<T&> { typedef T type; };
template<typename T> decltype(auto) move(T &&t) { return static_cast<typename remove_reference<T>::type&&>(t); }
auto s = [s(move(S()))] {};
