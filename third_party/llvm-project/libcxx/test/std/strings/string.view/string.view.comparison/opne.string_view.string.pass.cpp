//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string_view>

// template<class charT, class traits, class Allocator>
//   bool operator!=(const basic_string<charT, traits, Allocator> &lhs, basic_string_view<charT,traits> rhs);
// template<class charT, class traits, class Allocator>
//   bool operator!=(basic_string_view<charT,traits> lhs, const basic_string<charT, traits, Allocator> &rhs);

#include <string_view>
#include <string>
#include <cassert>

#include "test_macros.h"

template <class S>
void
test(const std::string &lhs, S rhs, bool x)
{
    assert((lhs != rhs) == x);
    assert((rhs != lhs) == x);
}

int main(int, char**)
{
    {
    typedef std::string_view S;
    test("", S(""), false);
    test("", S("abcde"), true);
    test("", S("abcdefghij"), true);
    test("", S("abcdefghijklmnopqrst"), true);
    test("abcde", S(""), true);
    test("abcde", S("abcde"), false);
    test("abcde", S("abcdefghij"), true);
    test("abcde", S("abcdefghijklmnopqrst"), true);
    test("abcdefghij", S(""), true);
    test("abcdefghij", S("abcde"), true);
    test("abcdefghij", S("abcdefghij"), false);
    test("abcdefghij", S("abcdefghijklmnopqrst"), true);
    test("abcdefghijklmnopqrst", S(""), true);
    test("abcdefghijklmnopqrst", S("abcde"), true);
    test("abcdefghijklmnopqrst", S("abcdefghij"), true);
    test("abcdefghijklmnopqrst", S("abcdefghijklmnopqrst"), false);
    }

  return 0;
}
