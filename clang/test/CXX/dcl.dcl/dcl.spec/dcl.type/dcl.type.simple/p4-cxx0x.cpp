// RUN: clang-cc -fsyntax-only -std=c++0x -verify %s

template<typename T, typename U>
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};

const int&& foo();
int i;
struct A { double x; };
const A* a = new A();

static_assert(is_same<decltype(foo()), const int&&>::value, "");
static_assert(is_same<decltype(i), int>::value, "");
static_assert(is_same<decltype(a->x), double>::value, "");
static_assert(is_same<decltype((a->x)), const double&>::value, "");
