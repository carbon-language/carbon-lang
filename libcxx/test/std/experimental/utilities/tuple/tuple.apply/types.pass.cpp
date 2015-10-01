//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <experimental/tuple>

// template <class F, class T> constexpr decltype(auto) apply(F &&, T &&)

// Test function types.

#include <experimental/tuple>
#include <array>
#include <utility>
#include <cassert>

// std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

namespace ex = std::experimental;

int count = 0;

void f_void_0() { ++count; }
void f_void_1(int i) { count += i; }
void f_void_2(int x, int y) { count += (x + y); }
void f_void_3(int x, int y, int z) { count += (x + y + z); }

int f_int_0() { return ++count; }
int f_int_1(int x) { return count += x; }
int f_int_2(int x, int y) { return count += (x + y); }
int f_int_3(int x, int y, int z) { return count += (x + y + z); }

struct A_void_0
{
    A_void_0() {}
    void operator()() { ++count; }
    void operator()() const { ++count; ++count; }
};

struct A_void_1
{
    A_void_1() {}
    void operator()(int x) { count += x; }
    void operator()(int x) const { count += x + 1; }
};

struct A_void_2
{
    A_void_2() {}
    void operator()(int x, int y) { count += (x + y); }
    void operator()(int x, int y) const { count += (x + y) + 1; }
};

struct A_void_3
{
    A_void_3() {}
    void operator()(int x, int y, int z) { count += (x + y + z); }
    void operator()(int x, int y, int z) const { count += (x + y + z) + 1; }
};


struct A_int_0
{
    A_int_0() {}
    int operator()() { return ++count; }
    int operator()() const { ++count; return ++count; }
};

struct A_int_1
{
    A_int_1() {}
    int operator()(int x) { return count += x; }
    int operator()(int x) const { return count += (x + 1); }

};

struct A_int_2
{
    A_int_2() {}
    int operator()(int x, int y) { return count += (x + y); }
    int operator()(int x, int y) const { return count += (x + y + 1); }
};

struct A_int_3
{
    A_int_3() {}
    int operator()(int x, int y, int z) { return count += (x + y + z); }
    int operator()(int x, int y, int z) const { return count += (x + y + z + 1); }
};


template <class Tuple>
void test_void_0()
{
    count = 0;
    // function
    {
        Tuple t{};
        ex::apply(f_void_0, t);
        assert(count == 1);
    }
    count = 0;
    // function pointer
    {
        Tuple t{};
        auto fp = &f_void_0;
        ex::apply(fp, t);
        assert(count == 1);
    }
    count = 0;
    // functor
    {
        Tuple t{};
        A_void_0 a;
        ex::apply(a, t);
        assert(count == 1);
    }
    count = 0;
    // const functor
    {
        Tuple t{};
        A_void_0 const a;
        ex::apply(a, t);
        assert(count == 2);
    }
}

template <class Tuple>
void test_void_1()
{
    count = 0;
    // function
    {
        Tuple t{1};
        ex::apply(f_void_1, t);
        assert(count == 1);
    }
    count = 0;
    // function pointer
    {
        Tuple t{2};
        void (*fp)(int) = f_void_1;
        ex::apply(fp, t);
        assert(count == 2);
    }
    count = 0;
    // functor
    {
        Tuple t{3};
        A_void_1 fn;
        ex::apply(fn, t);
        assert(count == 3);
    }
    count = 0;
    // const functor
    {
        Tuple t{4};
        A_void_1 const a;
        ex::apply(a, t);
        assert(count == 5);
    }
}

template <class Tuple>
void test_void_2()
{
    count = 0;
    // function
    {
        Tuple t{1, 2};
        ex::apply(f_void_2, t);
        assert(count == 3);
    }
    count = 0;
    // function pointer
    {
        Tuple t{2, 3};
        auto fp = &f_void_2;
        ex::apply(fp, t);
        assert(count == 5);
    }
    count = 0;
    // functor
    {
        Tuple t{3, 4};
        A_void_2 a;
        ex::apply(a, t);
        assert(count == 7);
    }
    count = 0;
    // const functor
    {
        Tuple t{4, 5};
        A_void_2 const a;
        ex::apply(a, t);
        assert(count == 10);
    }
}

template <class Tuple>
void test_void_3()
{
    count = 0;
    // function
    {
        Tuple t{1, 2, 3};
        ex::apply(f_void_3, t);
        assert(count == 6);
    }
    count = 0;
    // function pointer
    {
        Tuple t{2, 3, 4};
        auto fp = &f_void_3;
        ex::apply(fp, t);
        assert(count == 9);
    }
    count = 0;
    // functor
    {
        Tuple t{3, 4, 5};
        A_void_3 a;
        ex::apply(a, t);
        assert(count == 12);
    }
    count = 0;
    // const functor
    {
        Tuple t{4, 5, 6};
        A_void_3 const a;
        ex::apply(a, t);
        assert(count == 16);
    }
}



template <class Tuple>
void test_int_0()
{
    count = 0;
    // function
    {
        Tuple t{};
        assert(1 == ex::apply(f_int_0, t));
        assert(count == 1);
    }
    count = 0;
    // function pointer
    {
        Tuple t{};
        auto fp = &f_int_0;
        assert(1 == ex::apply(fp, t));
        assert(count == 1);
    }
    count = 0;
    // functor
    {
        Tuple t{};
        A_int_0 a;
        assert(1 == ex::apply(a, t));
        assert(count == 1);
    }
    count = 0;
    // const functor
    {
        Tuple t{};
        A_int_0 const a;
        assert(2 == ex::apply(a, t));
        assert(count == 2);
    }
}

template <class Tuple>
void test_int_1()
{
    count = 0;
    // function
    {
        Tuple t{1};
        assert(1 == ex::apply(f_int_1, t));
        assert(count == 1);
    }
    count = 0;
    // function pointer
    {
        Tuple t{2};
        int (*fp)(int) = f_int_1;
        assert(2 == ex::apply(fp, t));
        assert(count == 2);
    }
    count = 0;
    // functor
    {
        Tuple t{3};
        A_int_1 fn;
        assert(3 == ex::apply(fn, t));
        assert(count == 3);
    }
    count = 0;
    // const functor
    {
        Tuple t{4};
        A_int_1 const a;
        assert(5 == ex::apply(a, t));
        assert(count == 5);
    }
}

template <class Tuple>
void test_int_2()
{
    count = 0;
    // function
    {
        Tuple t{1, 2};
        assert(3 == ex::apply(f_int_2, t));
        assert(count == 3);
    }
    count = 0;
    // function pointer
    {
        Tuple t{2, 3};
        auto fp = &f_int_2;
        assert(5 == ex::apply(fp, t));
        assert(count == 5);
    }
    count = 0;
    // functor
    {
        Tuple t{3, 4};
        A_int_2 a;
        assert(7 == ex::apply(a, t));
        assert(count == 7);
    }
    count = 0;
    // const functor
    {
        Tuple t{4, 5};
        A_int_2 const a;
        assert(10 == ex::apply(a, t));
        assert(count == 10);
    }
}

template <class Tuple>
void test_int_3()
{
    count = 0;
    // function
    {
        Tuple t{1, 2, 3};
        assert(6 == ex::apply(f_int_3, t));
        assert(count == 6);
    }
    count = 0;
    // function pointer
    {
        Tuple t{2, 3, 4};
        auto fp = &f_int_3;
        assert(9 == ex::apply(fp, t));
        assert(count == 9);
    }
    count = 0;
    // functor
    {
        Tuple t{3, 4, 5};
        A_int_3 a;
        assert(12 == ex::apply(a, t));
        assert(count == 12);
    }
    count = 0;
    // const functor
    {
        Tuple t{4, 5, 6};
        A_int_3 const a;
        assert(16 == ex::apply(a, t));
        assert(count == 16);
    }
}

template <class Tuple>
void test_0()
{
    test_void_0<Tuple>();
    test_int_0<Tuple>();
}

template <class Tuple>
void test_1()
{
    test_void_1<Tuple>();
    test_int_1<Tuple>();
}

template <class Tuple>
void test_2()
{
    test_void_2<Tuple>();
    test_int_2<Tuple>();
}

template <class Tuple>
void test_3()
{
    test_void_3<Tuple>();
    test_int_3<Tuple>();
}

int main()
{
    test_0<std::tuple<>>();

    test_1<std::tuple<int>>();
    test_1<std::array<int, 1>>();

    test_2<std::tuple<int, int>>();
    test_2<std::pair<int, int>>();
    test_2<std::array<int, 2>>();

    test_3<std::tuple<int, int, int>>();
    test_3<std::array<int, 3>>();
}
