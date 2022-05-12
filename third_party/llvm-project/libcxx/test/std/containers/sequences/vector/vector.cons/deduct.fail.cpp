//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>
// UNSUPPORTED: c++03, c++11, c++14

// template <class InputIterator, class Allocator = allocator<typename iterator_traits<InputIterator>::value_type>>
//    vector(InputIterator, InputIterator, Allocator = Allocator())
//    -> vector<typename iterator_traits<InputIterator>::value_type, Allocator>;
//

#include <deque>
#include <iterator>
#include <cassert>
#include <cstddef>


int main(int, char**)
{
//  Test the explicit deduction guides

//  Test the implicit deduction guides
    {
//  vector (allocator &)
    std::vector vec((std::allocator<int>()));  // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'vector'}}
//  Note: The extra parens are necessary, since otherwise clang decides it is a function declaration.
//  Also, we can't use {} instead of parens, because that constructs a
//      deque<allocator<int>, allocator<allocator<int>>>
    }


  return 0;
}
