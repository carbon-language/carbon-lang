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

// constexpr span(pointer ptr, size_type count);
// Requires: [ptr, ptr + count) shall be a valid range.
//  If extent is not equal to dynamic_extent, then count shall be equal to extent.
//

#include <span>
#include <cassert>
#include <string>

#include "test_macros.h"


               int   arr[] = {1,2,3};
const          int  carr[] = {4,5,6};
      volatile int  varr[] = {7,8,9};
const volatile int cvarr[] = {1,3,5};

template<class T, size_t extent>
std::span<T, extent> createImplicitSpan(T* ptr, size_t len) {
    return {ptr, len}; // expected-error {{chosen constructor is explicit in copy-initialization}}
}

int main(int, char**)
{
//  We can't check that the size doesn't match - because that's a runtime property
//  std::span<int, 2>   s1(arr, 3);

//  Type wrong
    {
    std::span<float>    s1(arr, 3);   // expected-error {{no matching constructor for initialization of 'std::span<float>'}}
    std::span<float, 3> s2(arr, 3);   // expected-error {{no matching constructor for initialization of 'std::span<float, 3>'}}
    }

//  CV wrong (dynamically sized)
    {
    std::span<               int> s1{ carr, 3}; // expected-error {{no matching constructor for initialization of 'std::span<int>'}}
    std::span<               int> s2{ varr, 3}; // expected-error {{no matching constructor for initialization of 'std::span<int>'}}
    std::span<               int> s3{cvarr, 3}; // expected-error {{no matching constructor for initialization of 'std::span<int>'}}
    std::span<const          int> s4{ varr, 3}; // expected-error {{no matching constructor for initialization of 'std::span<const int>'}}
    std::span<const          int> s5{cvarr, 3}; // expected-error {{no matching constructor for initialization of 'std::span<const int>'}}
    std::span<      volatile int> s6{ carr, 3}; // expected-error {{no matching constructor for initialization of 'std::span<volatile int>'}}
    std::span<      volatile int> s7{cvarr, 3}; // expected-error {{no matching constructor for initialization of 'std::span<volatile int>'}}
    }

//  CV wrong (statically sized)
    {
    std::span<               int,3> s1{ carr, 3};   // expected-error {{no matching constructor for initialization of 'std::span<int, 3>'}}
    std::span<               int,3> s2{ varr, 3};   // expected-error {{no matching constructor for initialization of 'std::span<int, 3>'}}
    std::span<               int,3> s3{cvarr, 3};   // expected-error {{no matching constructor for initialization of 'std::span<int, 3>'}}
    std::span<const          int,3> s4{ varr, 3};   // expected-error {{no matching constructor for initialization of 'std::span<const int, 3>'}}
    std::span<const          int,3> s5{cvarr, 3};   // expected-error {{no matching constructor for initialization of 'std::span<const int, 3>'}}
    std::span<      volatile int,3> s6{ carr, 3};   // expected-error {{no matching constructor for initialization of 'std::span<volatile int, 3>'}}
    std::span<      volatile int,3> s7{cvarr, 3};   // expected-error {{no matching constructor for initialization of 'std::span<volatile int, 3>'}}
    }

// explicit constructor necessary
    {
    createImplicitSpan<int, 1>(arr, 1);
    }

  return 0;
}
