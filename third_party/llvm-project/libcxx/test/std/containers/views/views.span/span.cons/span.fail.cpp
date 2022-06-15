//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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

template<class T, size_t extent, size_t otherExtent>
std::span<T, extent> createImplicitSpan(std::span<T, otherExtent> s) {
    return {s}; // expected-error {{chosen constructor is explicit in copy-initialization}}
}

void checkCV ()
{
//  std::span<               int>   sp;
    std::span<const          int>  csp;
    std::span<      volatile int>  vsp;
    std::span<const volatile int> cvsp;

//  std::span<               int, 0>   sp0;
    std::span<const          int, 0>  csp0;
    std::span<      volatile int, 0>  vsp0;
    std::span<const volatile int, 0> cvsp0;

//  Try to remove const and/or volatile (dynamic -> dynamic)
    {
    std::span<               int> s1{ csp}; // expected-error {{no matching constructor for initialization of 'std::span<int>'}}
    std::span<               int> s2{ vsp}; // expected-error {{no matching constructor for initialization of 'std::span<int>'}}
    std::span<               int> s3{cvsp}; // expected-error {{no matching constructor for initialization of 'std::span<int>'}}

    std::span<const          int> s4{ vsp}; // expected-error {{no matching constructor for initialization of 'std::span<const int>'}}
    std::span<const          int> s5{cvsp}; // expected-error {{no matching constructor for initialization of 'std::span<const int>'}}

    std::span<      volatile int> s6{ csp}; // expected-error {{no matching constructor for initialization of 'std::span<volatile int>'}}
    std::span<      volatile int> s7{cvsp}; // expected-error {{no matching constructor for initialization of 'std::span<volatile int>'}}
    }

//  Try to remove const and/or volatile (static -> static)
    {
    std::span<               int, 0> s1{ csp0}; // expected-error {{no matching constructor for initialization of 'std::span<int, 0>'}}
    std::span<               int, 0> s2{ vsp0}; // expected-error {{no matching constructor for initialization of 'std::span<int, 0>'}}
    std::span<               int, 0> s3{cvsp0}; // expected-error {{no matching constructor for initialization of 'std::span<int, 0>'}}

    std::span<const          int, 0> s4{ vsp0}; // expected-error {{no matching constructor for initialization of 'std::span<const int, 0>'}}
    std::span<const          int, 0> s5{cvsp0}; // expected-error {{no matching constructor for initialization of 'std::span<const int, 0>'}}

    std::span<      volatile int, 0> s6{ csp0}; // expected-error {{no matching constructor for initialization of 'std::span<volatile int, 0>'}}
    std::span<      volatile int, 0> s7{cvsp0}; // expected-error {{no matching constructor for initialization of 'std::span<volatile int, 0>'}}
    }

//  Try to remove const and/or volatile (static -> dynamic)
    {
    std::span<               int> s1{ csp0}; // expected-error {{no matching constructor for initialization of 'std::span<int>'}}
    std::span<               int> s2{ vsp0}; // expected-error {{no matching constructor for initialization of 'std::span<int>'}}
    std::span<               int> s3{cvsp0}; // expected-error {{no matching constructor for initialization of 'std::span<int>'}}

    std::span<const          int> s4{ vsp0}; // expected-error {{no matching constructor for initialization of 'std::span<const int>'}}
    std::span<const          int> s5{cvsp0}; // expected-error {{no matching constructor for initialization of 'std::span<const int>'}}

    std::span<      volatile int> s6{ csp0}; // expected-error {{no matching constructor for initialization of 'std::span<volatile int>'}}
    std::span<      volatile int> s7{cvsp0}; // expected-error {{no matching constructor for initialization of 'std::span<volatile int>'}}
    }

//  Try to remove const and/or volatile (static -> static)
    {
    std::span<               int, 0> s1{ csp}; // expected-error {{no matching constructor for initialization of 'std::span<int, 0>'}}
    std::span<               int, 0> s2{ vsp}; // expected-error {{no matching constructor for initialization of 'std::span<int, 0>'}}
    std::span<               int, 0> s3{cvsp}; // expected-error {{no matching constructor for initialization of 'std::span<int, 0>'}}

    std::span<const          int, 0> s4{ vsp}; // expected-error {{no matching constructor for initialization of 'std::span<const int, 0>'}}
    std::span<const          int, 0> s5{cvsp}; // expected-error {{no matching constructor for initialization of 'std::span<const int, 0>'}}

    std::span<      volatile int, 0> s6{ csp}; // expected-error {{no matching constructor for initialization of 'std::span<volatile int, 0>'}}
    std::span<      volatile int, 0> s7{cvsp}; // expected-error {{no matching constructor for initialization of 'std::span<volatile int, 0>'}}
    }
}

int main(int, char**)
{
    std::span<int>      sp;
    std::span<int, 0>   sp0;

    std::span<float> s1{sp};    // expected-error {{no matching constructor for initialization of 'std::span<float>'}}
    std::span<float> s2{sp0};   // expected-error {{no matching constructor for initialization of 'std::span<float>'}}
    std::span<float, 0> s3{sp}; // expected-error {{no matching constructor for initialization of 'std::span<float, 0>'}}
    std::span<float, 0> s4{sp0};    // expected-error {{no matching constructor for initialization of 'std::span<float, 0>'}}

    checkCV();

    // explicit constructor necessary
    {
    createImplicitSpan<int, 1>(sp);
    }

  return 0;
}
