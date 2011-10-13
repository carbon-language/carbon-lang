// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

template <class T>
struct only
{
    only(T) {}

    template <class U>
    only(U)
    {
        static_assert(sizeof(U) == 0, "expected type failure");
    }
};

auto f() -> int
{
    return 0;
}

auto g(); // expected-error{{return without trailing return type}}

int h() -> int; // expected-error{{trailing return type must specify return type 'auto', not 'int'}}

int x;

template <class T>
auto i(T x) -> decltype(x)
{
    return x;
}

only<double> p1 = i(1.0);

template <class T>
struct X
{
    auto f(T x) -> T { return x; }

    template <class U>
    auto g(T x, U y) -> decltype(x + y)
    {
        return x + y;
    }

  template<typename U>
  struct nested {
    template <class V>
    auto h(T x, U y, V z) -> decltype(x + y + z)
    {
        return x + y + z;
    }
  };

  template<typename U>
  nested<U> get_nested();
};

X<int> xx;
only<int> p2 = xx.f(0L);
only<double> p3 = xx.g(0L, 1.0);
only<double> p4 = xx.get_nested<double>().h(0L, 1.0, 3.14f);
