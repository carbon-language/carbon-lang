// -*- C++ -*-
//===------------------------------ span ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

// constexpr span() noexcept;

#include <span>
#include <cassert>
#include <string>
#include <type_traits>

#include "test_macros.h"

void checkCV()
{
//  Types the same (dynamic sized)
    {
    std::span<               int> s1;
    std::span<const          int> s2;
    std::span<      volatile int> s3;
    std::span<const volatile int> s4;
    assert(s1.size() + s2.size() + s3.size() + s4.size() == 0);
    }

//  Types the same (static sized)
    {
    std::span<               int,0> s1;
    std::span<const          int,0> s2;
    std::span<      volatile int,0> s3;
    std::span<const volatile int,0> s4;
    assert(s1.size() + s2.size() + s3.size() + s4.size() == 0);
    }
}


template <typename T>
constexpr bool testConstexprSpan()
{
    std::span<const T>    s1;
    std::span<const T, 0> s2;
    return
        s1.data() == nullptr && s1.size() == 0
    &&  s2.data() == nullptr && s2.size() == 0;
}


template <typename T>
void testRuntimeSpan()
{
    ASSERT_NOEXCEPT(T{});
    std::span<const T>    s1;
    std::span<const T, 0> s2;
    assert(s1.data() == nullptr && s1.size() == 0);
    assert(s2.data() == nullptr && s2.size() == 0);
}


struct A{};

int main(int, char**)
{
    static_assert(testConstexprSpan<int>(),    "");
    static_assert(testConstexprSpan<long>(),   "");
    static_assert(testConstexprSpan<double>(), "");
    static_assert(testConstexprSpan<A>(),      "");

    testRuntimeSpan<int>();
    testRuntimeSpan<long>();
    testRuntimeSpan<double>();
    testRuntimeSpan<std::string>();
    testRuntimeSpan<A>();

    checkCV();

    static_assert( std::is_default_constructible_v<std::span<int, std::dynamic_extent>>, "");
    static_assert( std::is_default_constructible_v<std::span<int, 0>>, "");
    static_assert(!std::is_default_constructible_v<std::span<int, 2>>, "");

    return 0;
}
