//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// istreambuf_iterator

// bool equal(istreambuf_iterator<charT,traits>& b) const;

#include <iterator>
#include <sstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::istringstream inf1("abc");
        std::istringstream inf2("def");
        std::istreambuf_iterator<char> i1(inf1);
        std::istreambuf_iterator<char> i2(inf2);
        std::istreambuf_iterator<char> i3;
        std::istreambuf_iterator<char> i4;
        std::istreambuf_iterator<char> i5(nullptr);

        assert( i1.equal(i1));
        assert( i1.equal(i2));
        assert(!i1.equal(i3));
        assert(!i1.equal(i4));
        assert(!i1.equal(i5));

        assert( i2.equal(i1));
        assert( i2.equal(i2));
        assert(!i2.equal(i3));
        assert(!i2.equal(i4));
        assert(!i2.equal(i5));

        assert(!i3.equal(i1));
        assert(!i3.equal(i2));
        assert( i3.equal(i3));
        assert( i3.equal(i4));
        assert( i3.equal(i5));

        assert(!i4.equal(i1));
        assert(!i4.equal(i2));
        assert( i4.equal(i3));
        assert( i4.equal(i4));
        assert( i4.equal(i5));

        assert(!i5.equal(i1));
        assert(!i5.equal(i2));
        assert( i5.equal(i3));
        assert( i5.equal(i4));
        assert( i5.equal(i5));
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::wistringstream inf1(L"abc");
        std::wistringstream inf2(L"def");
        std::istreambuf_iterator<wchar_t> i1(inf1);
        std::istreambuf_iterator<wchar_t> i2(inf2);
        std::istreambuf_iterator<wchar_t> i3;
        std::istreambuf_iterator<wchar_t> i4;
        std::istreambuf_iterator<wchar_t> i5(nullptr);

        assert( i1.equal(i1));
        assert( i1.equal(i2));
        assert(!i1.equal(i3));
        assert(!i1.equal(i4));
        assert(!i1.equal(i5));

        assert( i2.equal(i1));
        assert( i2.equal(i2));
        assert(!i2.equal(i3));
        assert(!i2.equal(i4));
        assert(!i2.equal(i5));

        assert(!i3.equal(i1));
        assert(!i3.equal(i2));
        assert( i3.equal(i3));
        assert( i3.equal(i4));
        assert( i3.equal(i5));

        assert(!i4.equal(i1));
        assert(!i4.equal(i2));
        assert( i4.equal(i3));
        assert( i4.equal(i4));
        assert( i4.equal(i5));

        assert(!i5.equal(i1));
        assert(!i5.equal(i2));
        assert( i5.equal(i3));
        assert( i5.equal(i4));
        assert( i5.equal(i5));
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS

  return 0;
}
