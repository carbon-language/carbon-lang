//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <class... UTypes> tuple(const tuple<UTypes...>& u);

// UNSUPPORTED: c++03

#include <tuple>
#include <utility>
#include <string>
#include <cassert>

#include "test_macros.h"

struct Explicit {
  int value;
  explicit Explicit(int x) : value(x) {}
};

struct Implicit {
  int value;
  Implicit(int x) : value(x) {}
};

struct ExplicitTwo {
    ExplicitTwo() {}
    ExplicitTwo(ExplicitTwo const&) {}
    ExplicitTwo(ExplicitTwo &&) {}

    template <class T, class = typename std::enable_if<!std::is_same<T, ExplicitTwo>::value>::type>
    explicit ExplicitTwo(T) {}
};

struct B
{
    int id_;

    explicit B(int i) : id_(i) {}
};

struct D
    : B
{
    explicit D(int i) : B(i) {}
};

#if TEST_STD_VER > 11

struct A
{
    int id_;

    constexpr A(int i) : id_(i) {}
    friend constexpr bool operator==(const A& x, const A& y) {return x.id_ == y.id_;}
};

struct C
{
    int id_;

    constexpr explicit C(int i) : id_(i) {}
    friend constexpr bool operator==(const C& x, const C& y) {return x.id_ == y.id_;}
};

#endif

int main(int, char**)
{
    {
        typedef std::tuple<long> T0;
        typedef std::tuple<long long> T1;
        T0 t0(2);
        T1 t1 = t0;
        assert(std::get<0>(t1) == 2);
    }
#if TEST_STD_VER > 11
    {
        typedef std::tuple<int> T0;
        typedef std::tuple<A> T1;
        constexpr T0 t0(2);
        constexpr T1 t1 = t0;
        static_assert(std::get<0>(t1) == 2, "");
    }
    {
        typedef std::tuple<int> T0;
        typedef std::tuple<C> T1;
        constexpr T0 t0(2);
        constexpr T1 t1{t0};
        static_assert(std::get<0>(t1) == C(2), "");
    }
#endif
    {
        typedef std::tuple<long, char> T0;
        typedef std::tuple<long long, int> T1;
        T0 t0(2, 'a');
        T1 t1 = t0;
        assert(std::get<0>(t1) == 2);
        assert(std::get<1>(t1) == int('a'));
    }
    {
        typedef std::tuple<long, char, D> T0;
        typedef std::tuple<long long, int, B> T1;
        T0 t0(2, 'a', D(3));
        T1 t1 = t0;
        assert(std::get<0>(t1) == 2);
        assert(std::get<1>(t1) == int('a'));
        assert(std::get<2>(t1).id_ == 3);
    }
    {
        D d(3);
        typedef std::tuple<long, char, D&> T0;
        typedef std::tuple<long long, int, B&> T1;
        T0 t0(2, 'a', d);
        T1 t1 = t0;
        d.id_ = 2;
        assert(std::get<0>(t1) == 2);
        assert(std::get<1>(t1) == int('a'));
        assert(std::get<2>(t1).id_ == 2);
    }
    {
        typedef std::tuple<long, char, int> T0;
        typedef std::tuple<long long, int, B> T1;
        T0 t0(2, 'a', 3);
        T1 t1(t0);
        assert(std::get<0>(t1) == 2);
        assert(std::get<1>(t1) == int('a'));
        assert(std::get<2>(t1).id_ == 3);
    }
    {
        const std::tuple<int> t1(42);
        std::tuple<Explicit> t2(t1);
        assert(std::get<0>(t2).value == 42);
    }
    {
        const std::tuple<int> t1(42);
        std::tuple<Implicit> t2 = t1;
        assert(std::get<0>(t2).value == 42);
    }
    {
        static_assert(std::is_convertible<ExplicitTwo&&, ExplicitTwo>::value, "");
        static_assert(std::is_convertible<std::tuple<ExplicitTwo&&>&&, const std::tuple<ExplicitTwo>&>::value, "");

        ExplicitTwo e;
        std::tuple<ExplicitTwo> t = std::tuple<ExplicitTwo&&>(std::move(e));
        ((void)t);
    }
  return 0;
}
