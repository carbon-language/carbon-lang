//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// const_iterator cend() const;

#include <string>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
void
test(const S& s)
{
    typename S::const_iterator ce = s.cend();
    assert(ce == s.end());
}

int main(int, char**)
{
    {
    typedef std::string S;
    test(S());
    test(S("123"));
    }
#if TEST_STD_VER >= 11
    {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    test(S());
    test(S("123"));
    }
#endif

  return 0;
}
