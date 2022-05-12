//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// tuple& operator=(const tuple& u);

// UNSUPPORTED: c++03

#include <tuple>
#include <memory>
#include <string>
#include <cassert>

#include "test_macros.h"

struct NonAssignable {
  NonAssignable& operator=(NonAssignable const&) = delete;
  NonAssignable& operator=(NonAssignable&&) = delete;
};
struct CopyAssignable {
  CopyAssignable& operator=(CopyAssignable const&) = default;
  CopyAssignable& operator=(CopyAssignable &&) = delete;
};
static_assert(std::is_copy_assignable<CopyAssignable>::value, "");
struct MoveAssignable {
  MoveAssignable& operator=(MoveAssignable const&) = delete;
  MoveAssignable& operator=(MoveAssignable&&) = default;
};
struct NothrowCopyAssignable {
  NothrowCopyAssignable& operator=(NothrowCopyAssignable const&) noexcept { return *this; }
};
struct PotentiallyThrowingCopyAssignable {
  PotentiallyThrowingCopyAssignable& operator=(PotentiallyThrowingCopyAssignable const&) { return *this; }
};

struct CopyAssignableInt {
  CopyAssignableInt& operator=(int&) { return *this; }
};

TEST_CONSTEXPR_CXX20
bool test()
{
    {
        typedef std::tuple<> T;
        T t0;
        T t;
        t = t0;
    }
    {
        typedef std::tuple<int> T;
        T t0(2);
        T t;
        t = t0;
        assert(std::get<0>(t) == 2);
    }
    {
        typedef std::tuple<int, char> T;
        T t0(2, 'a');
        T t;
        t = t0;
        assert(std::get<0>(t) == 2);
        assert(std::get<1>(t) == 'a');
    }
    {
        // test reference assignment.
        using T = std::tuple<int&, int&&>;
        int x = 42;
        int y = 100;
        int x2 = -1;
        int y2 = 500;
        T t(x, std::move(y));
        T t2(x2, std::move(y2));
        t = t2;
        assert(std::get<0>(t) == x2);
        assert(&std::get<0>(t) == &x);
        assert(std::get<1>(t) == y2);
        assert(&std::get<1>(t) == &y);
    }

    return true;
}

int main(int, char**)
{
    test();
#if TEST_STD_VER >= 20
    static_assert(test());
#endif

    {
        // cannot be constexpr because of std::string
        typedef std::tuple<int, char, std::string> T;
        const T t0(2, 'a', "some text");
        T t;
        t = t0;
        assert(std::get<0>(t) == 2);
        assert(std::get<1>(t) == 'a');
        assert(std::get<2>(t) == "some text");
    }
    {
        // test that the implicitly generated copy assignment operator
        // is properly deleted
        using T = std::tuple<std::unique_ptr<int>>;
        static_assert(!std::is_copy_assignable<T>::value, "");
    }
    {
        using T = std::tuple<int, NonAssignable>;
        static_assert(!std::is_copy_assignable<T>::value, "");
    }
    {
        using T = std::tuple<int, CopyAssignable>;
        static_assert(std::is_copy_assignable<T>::value, "");
    }
    {
        using T = std::tuple<int, MoveAssignable>;
        static_assert(!std::is_copy_assignable<T>::value, "");
    }
    {
        using T = std::tuple<int, int, int>;
        using P = std::pair<int, int>;
        static_assert(!std::is_assignable<T&, P>::value, "");
    }
    {
        // test const requirement
        using T = std::tuple<CopyAssignableInt, CopyAssignableInt>;
        using P = std::pair<int, int>;
        static_assert(!std::is_assignable<T&, P const>::value, "");
    }
    {
        using T = std::tuple<int, MoveAssignable>;
        using P = std::pair<int, MoveAssignable>;
        static_assert(!std::is_assignable<T&, P&>::value, "");
    }
    {
        using T = std::tuple<NothrowCopyAssignable, int>;
        static_assert(std::is_nothrow_copy_assignable<T>::value, "");
    }
    {
        using T = std::tuple<PotentiallyThrowingCopyAssignable, int>;
        static_assert(std::is_copy_assignable<T>::value, "");
        static_assert(!std::is_nothrow_copy_assignable<T>::value, "");
    }

    return 0;
}
