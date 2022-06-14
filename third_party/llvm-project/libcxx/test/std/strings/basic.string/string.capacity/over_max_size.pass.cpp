//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11}}

// Prior to http://llvm.org/D123580, there was a bug with how the max_size()
// was calculated. That was inlined into some functions in the dylib, which leads
// to failures when running this test against an older system dylib.
// XFAIL: use_system_cxx_lib && target=arm64-apple-macosx{{11.0|12.0}}

// <string>

// size_type max_size() const; // constexpr since C++20

#include <string>
#include <cassert>
#include <stdexcept>

#include "test_macros.h"
#include "min_allocator.h"

template <class S>
void
test(const S& s)
{
    assert(s.max_size() >= s.size());
    S s2(s);
    const size_t sz = s2.max_size() + 1;
    try { s2.resize(sz, 'x'); }
    catch ( const std::length_error & ) { return ; }
    assert ( false );
}

bool test() {
  {
    typedef std::string S;
    test(S());
    test(S("123"));
    test(S("12345678901234567890123456789012345678901234567890"));
  }
#if TEST_STD_VER >= 11
  {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    test(S());
    test(S("123"));
    test(S("12345678901234567890123456789012345678901234567890"));
  }
#endif

  return true;
}

int main(int, char**)
{
  test();

  return 0;
}
