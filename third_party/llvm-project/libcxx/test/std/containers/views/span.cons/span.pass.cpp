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

// template<class OtherElementType, size_t OtherExtent>
//    constexpr span(const span<OtherElementType, OtherExtent>& s) noexcept;
//
//  Remarks: This constructor shall not participate in overload resolution unless:
//      Extent == dynamic_extent || Extent == OtherExtent is true, and
//      OtherElementType(*)[] is convertible to ElementType(*)[].


#include <span>
#include <cassert>
#include <string>

#include "test_macros.h"

void checkCV()
{
    std::span<               int>   sp;
//  std::span<const          int>  csp;
    std::span<      volatile int>  vsp;
//  std::span<const volatile int> cvsp;

    std::span<               int, 0>   sp0;
//  std::span<const          int, 0>  csp0;
    std::span<      volatile int, 0>  vsp0;
//  std::span<const volatile int, 0> cvsp0;

//  dynamic -> dynamic
    {
        std::span<const          int> s1{  sp}; // a span<const          int> pointing at int.
        std::span<      volatile int> s2{  sp}; // a span<      volatile int> pointing at int.
        std::span<const volatile int> s3{  sp}; // a span<const volatile int> pointing at int.
        std::span<const volatile int> s4{ vsp}; // a span<const volatile int> pointing at volatile int.
        assert(s1.size() + s2.size() + s3.size() + s4.size() == 0);
    }

//  static -> static
    {
        std::span<const          int, 0> s1{  sp0}; // a span<const          int> pointing at int.
        std::span<      volatile int, 0> s2{  sp0}; // a span<      volatile int> pointing at int.
        std::span<const volatile int, 0> s3{  sp0}; // a span<const volatile int> pointing at int.
        std::span<const volatile int, 0> s4{ vsp0}; // a span<const volatile int> pointing at volatile int.
        assert(s1.size() + s2.size() + s3.size() + s4.size() == 0);
    }

//  static -> dynamic
    {
        std::span<const          int> s1{  sp0};    // a span<const          int> pointing at int.
        std::span<      volatile int> s2{  sp0};    // a span<      volatile int> pointing at int.
        std::span<const volatile int> s3{  sp0};    // a span<const volatile int> pointing at int.
        std::span<const volatile int> s4{ vsp0};    // a span<const volatile int> pointing at volatile int.
        assert(s1.size() + s2.size() + s3.size() + s4.size() == 0);
    }

//  dynamic -> static (not allowed)
}


template <typename T>
constexpr bool testConstexprSpan()
{
    std::span<T>    s0{};
    std::span<T, 0> s1{};
    std::span<T>    s2(s1); // static -> dynamic
    ASSERT_NOEXCEPT(std::span<T>   {s0});
    ASSERT_NOEXCEPT(std::span<T, 0>{s1});
    ASSERT_NOEXCEPT(std::span<T>   {s1});

    return
        s1.data() == nullptr && s1.size() == 0
    &&  s2.data() == nullptr && s2.size() == 0;
}


template <typename T>
void testRuntimeSpan()
{
    std::span<T>    s0{};
    std::span<T, 0> s1{};
    std::span<T>    s2(s1); // static -> dynamic
    ASSERT_NOEXCEPT(std::span<T>   {s0});
    ASSERT_NOEXCEPT(std::span<T, 0>{s1});
    ASSERT_NOEXCEPT(std::span<T>   {s1});

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

  return 0;
}
