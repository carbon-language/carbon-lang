//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>
// UNSUPPORTED: c++03, c++11, c++14

// template<class InputIterator>
//   basic_string(InputIterator begin, InputIterator end,
//   const Allocator& a = Allocator());

// template<class charT,
//          class traits,
//          class Allocator = allocator<charT>
//          >
// basic_string(basic_string_view<charT, traits>,
//                typename see below::size_type,
//                typename see below::size_type,
//                const Allocator& = Allocator())
//   -> basic_string<charT, traits, Allocator>;
//
//  A size_type parameter type in a basic_string deduction guide refers to the size_type
//  member type of the type deduced by the deduction guide.
//
//  The deduction guide shall not participate in overload resolution if Allocator
//  is a type that does not qualify as an allocator.

#include <string>
#include <string_view>
#include <iterator>
#include <cassert>
#include <cstddef>

int main(int, char**)
{
    {
    std::string_view sv = "12345678901234";
    std::basic_string s1{sv, 0, 4, 23}; // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'basic_string'}}
    }

  return 0;
}
