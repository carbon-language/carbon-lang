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

// Test with different ref/ptr/cv qualified argument types.

#include <experimental/tuple>
#include <array>
#include <utility>
#include <cassert>

namespace ex = std::experimental;

int call_with_value(int x, int y) { return (x + y); }
int call_with_ref(int & x, int & y) { return (x + y); }
int call_with_const_ref(int const & x, int const & y) { return (x + y); }
int call_with_rvalue_ref(int && x, int && y) { return (x + y); }
int call_with_pointer(int * x, int * y) { return (*x + *y); }
int call_with_const_pointer(int const* x, int const * y) { return (*x + *y); }


template <class Tuple>
void test_values()
{
    {
        Tuple t{1, 2};
        assert(3 == ex::apply(call_with_value, t));
    }
    {
        Tuple t{2, 2};
        assert(4 == ex::apply(call_with_ref, t));
    }
    {
        Tuple t{2, 3};
        assert(5 == ex::apply(call_with_const_ref, t));
    }
    {
        Tuple t{3, 3};
        assert(6 == ex::apply(call_with_rvalue_ref, static_cast<Tuple &&>(t)));
    }
    {
        Tuple const t{4, 4};
        assert(8 == ex::apply(call_with_value, t));
    }
    {
        Tuple const t{4, 5};
        assert(9 == ex::apply(call_with_const_ref, t));
    }
}

template <class Tuple>
void test_refs()
{
    int x = 0;
    int y = 0;
    {
        x = 1; y = 2;
        Tuple t{x, y};
        assert(3 == ex::apply(call_with_value, t));
    }
    {
        x = 2; y = 2;
        Tuple t{x, y};
        assert(4 == ex::apply(call_with_ref, t));
    }
    {
        x = 2; y = 3;
        Tuple t{x, y};
        assert(5 == ex::apply(call_with_const_ref, t));
    }
    {
        x = 3; y = 3;
        Tuple const t{x, y};
        assert(6 == ex::apply(call_with_value, t));
    }
    {
        x = 3; y = 4;
        Tuple const t{x, y};
        assert(7 == ex::apply(call_with_const_ref, t));
    }
}

template <class Tuple>
void test_const_refs()
{
    int x = 0;
    int y = 0;
    {
        x = 1; y = 2;
        Tuple t{x, y};
        assert(3 == ex::apply(call_with_value, t));
    }
    {
        x = 2; y = 3;
        Tuple t{x, y};
        assert(5 == ex::apply(call_with_const_ref, t));
    }
    {
        x = 3; y = 3;
        Tuple const t{x, y};
        assert(6 == ex::apply(call_with_value, t));
    }
    {
        x = 3; y = 4;
        Tuple const t{x, y};
        assert(7 == ex::apply(call_with_const_ref, t));
    }
}


template <class Tuple>
void test_pointer()
{
    int x = 0;
    int y = 0;
    {
        x = 2; y = 2;
        Tuple t{&x, &y};
        assert(4 == ex::apply(call_with_pointer, t));
    }
    {
        x = 2; y = 3;
        Tuple t{&x, &y};
        assert(5 == ex::apply(call_with_const_pointer, t));
    }
    {
        x = 3; y = 4;
        Tuple const t{&x, &y};
        assert(7 == ex::apply(call_with_const_pointer, t));
    }
}


template <class Tuple>
void test_const_pointer()
{
    int x = 0;
    int y = 0;
    {
        x = 2; y = 3;
        Tuple t{&x, &y};
        assert(5 == ex::apply(call_with_const_pointer, t));
    }
    {
        x = 3; y = 4;
        Tuple const t{&x, &y};
        assert(7 == ex::apply(call_with_const_pointer, t));
    }
}


int main()
{
    test_values<std::tuple<int, int>>();
    test_values<std::pair<int, int>>();
    test_values<std::array<int, 2>>();

    test_refs<std::tuple<int &, int &>>();
    test_refs<std::pair<int &, int &>>();

    test_const_refs<std::tuple<int const &, int const &>>();
    test_const_refs<std::pair<int const &, int const &>>();

    test_pointer<std::tuple<int *, int *>>();
    test_pointer<std::pair<int *, int *>>();
    test_pointer<std::array<int *, 2>>();

    test_const_pointer<std::tuple<int const *, int const *>>();
    test_const_pointer<std::pair<int const *, int const *>>();
    test_const_pointer<std::array<int const *, 2>>();
}
