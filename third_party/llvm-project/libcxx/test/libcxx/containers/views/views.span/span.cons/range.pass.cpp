//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

//  template<class Container>
//    constexpr explicit(Extent != dynamic_extent) span(Container&);
//  template<class Container>
//    constexpr explicit(Extent != dynamic_extent) span(Container const&);

// This test checks for libc++'s non-conforming temporary extension to std::span
// to support construction from containers that look like contiguous ranges.
//
// This extension is only supported when we don't ship <ranges>, and we can
// remove it once we get rid of _LIBCPP_HAS_NO_INCOMPLETE_RANGES.

#include <span>
#include <cassert>
#include <string>
#include <vector>

#include "test_macros.h"

//  Look ma - I'm a container!
template <typename T>
struct IsAContainer {
    constexpr IsAContainer() : v_{} {}
    constexpr size_t size() const {return 1;}
    constexpr       T *data() {return &v_;}
    constexpr const T *data() const {return &v_;}
    constexpr       T *begin() {return &v_;}
    constexpr const T *begin() const {return &v_;}
    constexpr       T *end() {return &v_ + 1;}
    constexpr const T *end() const {return &v_ + 1;}

    constexpr T const *getV() const {return &v_;} // for checking
    T v_;
};


void checkCV()
{
    std::vector<int> v  = {1,2,3};

//  Types the same
    {
    std::span<               int> s1{v};    // a span<               int> pointing at int.
    }

//  types different
    {
    std::span<const          int> s1{v};    // a span<const          int> pointing at int.
    std::span<      volatile int> s2{v};    // a span<      volatile int> pointing at int.
    std::span<      volatile int> s3{v};    // a span<      volatile int> pointing at const int.
    std::span<const volatile int> s4{v};    // a span<const volatile int> pointing at int.
    }

//  Constructing a const view from a temporary
    {
    std::span<const int>    s1{IsAContainer<int>()};
    std::span<const int>    s3{std::vector<int>()};
    (void) s1;
    (void) s3;
    }
}


template <typename T>
constexpr bool testConstexprSpan()
{
    constexpr IsAContainer<const T> val{};
    std::span<const T> s1{val};
    return s1.data() == val.getV() && s1.size() == 1;
}

template <typename T>
constexpr bool testConstexprSpanStatic()
{
    constexpr IsAContainer<const T> val{};
    std::span<const T, 1> s1{val};
    return s1.data() == val.getV() && s1.size() == 1;
}

template <typename T>
void testRuntimeSpan()
{
    IsAContainer<T> val{};
    const IsAContainer<T> cVal;
    std::span<T>       s1{val};
    std::span<const T> s2{cVal};
    assert(s1.data() == val.getV()  && s1.size() == 1);
    assert(s2.data() == cVal.getV() && s2.size() == 1);
}

template <typename T>
void testRuntimeSpanStatic()
{
    IsAContainer<T> val{};
    const IsAContainer<T> cVal;
    std::span<T, 1>       s1{val};
    std::span<const T, 1> s2{cVal};
    assert(s1.data() == val.getV()  && s1.size() == 1);
    assert(s2.data() == cVal.getV() && s2.size() == 1);
}

struct A{};

int main(int, char**)
{
    static_assert(testConstexprSpan<int>(),    "");
    static_assert(testConstexprSpan<long>(),   "");
    static_assert(testConstexprSpan<double>(), "");
    static_assert(testConstexprSpan<A>(),      "");

    static_assert(testConstexprSpanStatic<int>(),    "");
    static_assert(testConstexprSpanStatic<long>(),   "");
    static_assert(testConstexprSpanStatic<double>(), "");
    static_assert(testConstexprSpanStatic<A>(),      "");

    testRuntimeSpan<int>();
    testRuntimeSpan<long>();
    testRuntimeSpan<double>();
    testRuntimeSpan<std::string>();
    testRuntimeSpan<A>();

    testRuntimeSpanStatic<int>();
    testRuntimeSpanStatic<long>();
    testRuntimeSpanStatic<double>();
    testRuntimeSpanStatic<std::string>();
    testRuntimeSpanStatic<A>();

    checkCV();

    return 0;
}
