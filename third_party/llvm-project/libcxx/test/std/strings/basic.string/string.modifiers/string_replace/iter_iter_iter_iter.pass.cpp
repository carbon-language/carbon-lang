//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<class InputIterator>
//   basic_string&
//   replace(const_iterator i1, const_iterator i2, InputIterator j1, InputIterator j2);

#include <string>
#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"
#include "test_iterators.h"

template <class S, class It>
TEST_CONSTEXPR_CXX20 void
test(S s, typename S::size_type pos1, typename S::size_type n1, It f, It l, S expected)
{
    typename S::size_type old_size = s.size();
    typename S::const_iterator first = s.begin() + pos1;
    typename S::const_iterator last = s.begin() + pos1 + n1;
    typename S::size_type xlen = last - first;
    s.replace(first, last, f, l);
    LIBCPP_ASSERT(s.__invariants());
    assert(s == expected);
    typename S::size_type rlen = std::distance(f, l);
    assert(s.size() == old_size - xlen + rlen);
}

#ifndef TEST_HAS_NO_EXCEPTIONS
struct Widget { operator char() const { throw 42; } };

template <class S, class It>
TEST_CONSTEXPR_CXX20 void
test_exceptions(S s, typename S::size_type pos1, typename S::size_type n1, It f, It l)
{
    typename S::const_iterator first = s.begin() + pos1;
    typename S::const_iterator last = s.begin() + pos1 + n1;

    S original = s;
    typename S::iterator begin = s.begin();
    typename S::iterator end = s.end();

    try {
        s.replace(first, last, f, l);
        assert(false);
    } catch (...) {}

    // Part of "no effects" is that iterators and pointers
    // into the string must not have been invalidated.
    LIBCPP_ASSERT(s.__invariants());
    assert(s == original);
    assert(s.begin() == begin);
    assert(s.end() == end);
}
#endif

const char* str = "12345678901234567890";

template <class S>
TEST_CONSTEXPR_CXX20 bool test0()
{
    test(S(""), 0, 0, str, str+0, S(""));
    test(S(""), 0, 0, str, str+0, S(""));
    test(S(""), 0, 0, str, str+1, S("1"));
    test(S(""), 0, 0, str, str+2, S("12"));
    test(S(""), 0, 0, str, str+4, S("1234"));
    test(S(""), 0, 0, str, str+5, S("12345"));
    test(S(""), 0, 0, str, str+0, S(""));
    test(S(""), 0, 0, str, str+1, S("1"));
    test(S(""), 0, 0, str, str+5, S("12345"));
    test(S(""), 0, 0, str, str+9, S("123456789"));
    test(S(""), 0, 0, str, str+10, S("1234567890"));
    test(S(""), 0, 0, str, str+0, S(""));
    test(S(""), 0, 0, str, str+1, S("1"));
    test(S(""), 0, 0, str, str+10, S("1234567890"));
    test(S(""), 0, 0, str, str+19, S("1234567890123456789"));
    test(S(""), 0, 0, str, str+20, S("12345678901234567890"));
    test(S("abcde"), 0, 0, str, str+0, S("abcde"));
    test(S("abcde"), 0, 0, str, str+0, S("abcde"));
    test(S("abcde"), 0, 0, str, str+1, S("1abcde"));
    test(S("abcde"), 0, 0, str, str+2, S("12abcde"));
    test(S("abcde"), 0, 0, str, str+4, S("1234abcde"));
    test(S("abcde"), 0, 0, str, str+5, S("12345abcde"));
    test(S("abcde"), 0, 0, str, str+0, S("abcde"));
    test(S("abcde"), 0, 0, str, str+1, S("1abcde"));
    test(S("abcde"), 0, 0, str, str+5, S("12345abcde"));
    test(S("abcde"), 0, 0, str, str+9, S("123456789abcde"));
    test(S("abcde"), 0, 0, str, str+10, S("1234567890abcde"));
    test(S("abcde"), 0, 0, str, str+0, S("abcde"));
    test(S("abcde"), 0, 0, str, str+1, S("1abcde"));
    test(S("abcde"), 0, 0, str, str+10, S("1234567890abcde"));
    test(S("abcde"), 0, 0, str, str+19, S("1234567890123456789abcde"));
    test(S("abcde"), 0, 0, str, str+20, S("12345678901234567890abcde"));
    test(S("abcde"), 0, 1, str, str+0, S("bcde"));
    test(S("abcde"), 0, 1, str, str+0, S("bcde"));
    test(S("abcde"), 0, 1, str, str+1, S("1bcde"));
    test(S("abcde"), 0, 1, str, str+2, S("12bcde"));
    test(S("abcde"), 0, 1, str, str+4, S("1234bcde"));
    test(S("abcde"), 0, 1, str, str+5, S("12345bcde"));
    test(S("abcde"), 0, 1, str, str+0, S("bcde"));
    test(S("abcde"), 0, 1, str, str+1, S("1bcde"));
    test(S("abcde"), 0, 1, str, str+5, S("12345bcde"));
    test(S("abcde"), 0, 1, str, str+9, S("123456789bcde"));
    test(S("abcde"), 0, 1, str, str+10, S("1234567890bcde"));
    test(S("abcde"), 0, 1, str, str+0, S("bcde"));
    test(S("abcde"), 0, 1, str, str+1, S("1bcde"));
    test(S("abcde"), 0, 1, str, str+10, S("1234567890bcde"));
    test(S("abcde"), 0, 1, str, str+19, S("1234567890123456789bcde"));
    test(S("abcde"), 0, 1, str, str+20, S("12345678901234567890bcde"));
    test(S("abcde"), 0, 2, str, str+0, S("cde"));
    test(S("abcde"), 0, 2, str, str+0, S("cde"));
    test(S("abcde"), 0, 2, str, str+1, S("1cde"));
    test(S("abcde"), 0, 2, str, str+2, S("12cde"));
    test(S("abcde"), 0, 2, str, str+4, S("1234cde"));
    test(S("abcde"), 0, 2, str, str+5, S("12345cde"));
    test(S("abcde"), 0, 2, str, str+0, S("cde"));
    test(S("abcde"), 0, 2, str, str+1, S("1cde"));
    test(S("abcde"), 0, 2, str, str+5, S("12345cde"));
    test(S("abcde"), 0, 2, str, str+9, S("123456789cde"));
    test(S("abcde"), 0, 2, str, str+10, S("1234567890cde"));
    test(S("abcde"), 0, 2, str, str+0, S("cde"));
    test(S("abcde"), 0, 2, str, str+1, S("1cde"));
    test(S("abcde"), 0, 2, str, str+10, S("1234567890cde"));
    test(S("abcde"), 0, 2, str, str+19, S("1234567890123456789cde"));
    test(S("abcde"), 0, 2, str, str+20, S("12345678901234567890cde"));
    test(S("abcde"), 0, 4, str, str+0, S("e"));
    test(S("abcde"), 0, 4, str, str+0, S("e"));
    test(S("abcde"), 0, 4, str, str+1, S("1e"));
    test(S("abcde"), 0, 4, str, str+2, S("12e"));
    test(S("abcde"), 0, 4, str, str+4, S("1234e"));
    test(S("abcde"), 0, 4, str, str+5, S("12345e"));
    test(S("abcde"), 0, 4, str, str+0, S("e"));
    test(S("abcde"), 0, 4, str, str+1, S("1e"));
    test(S("abcde"), 0, 4, str, str+5, S("12345e"));
    test(S("abcde"), 0, 4, str, str+9, S("123456789e"));
    test(S("abcde"), 0, 4, str, str+10, S("1234567890e"));
    test(S("abcde"), 0, 4, str, str+0, S("e"));
    test(S("abcde"), 0, 4, str, str+1, S("1e"));
    test(S("abcde"), 0, 4, str, str+10, S("1234567890e"));
    test(S("abcde"), 0, 4, str, str+19, S("1234567890123456789e"));
    test(S("abcde"), 0, 4, str, str+20, S("12345678901234567890e"));
    test(S("abcde"), 0, 5, str, str+0, S(""));
    test(S("abcde"), 0, 5, str, str+0, S(""));
    test(S("abcde"), 0, 5, str, str+1, S("1"));
    test(S("abcde"), 0, 5, str, str+2, S("12"));
    test(S("abcde"), 0, 5, str, str+4, S("1234"));
    test(S("abcde"), 0, 5, str, str+5, S("12345"));
    test(S("abcde"), 0, 5, str, str+0, S(""));
    test(S("abcde"), 0, 5, str, str+1, S("1"));
    test(S("abcde"), 0, 5, str, str+5, S("12345"));
    test(S("abcde"), 0, 5, str, str+9, S("123456789"));
    test(S("abcde"), 0, 5, str, str+10, S("1234567890"));
    test(S("abcde"), 0, 5, str, str+0, S(""));
    test(S("abcde"), 0, 5, str, str+1, S("1"));
    test(S("abcde"), 0, 5, str, str+10, S("1234567890"));
    test(S("abcde"), 0, 5, str, str+19, S("1234567890123456789"));
    test(S("abcde"), 0, 5, str, str+20, S("12345678901234567890"));
    test(S("abcde"), 1, 0, str, str+0, S("abcde"));
    test(S("abcde"), 1, 0, str, str+0, S("abcde"));
    test(S("abcde"), 1, 0, str, str+1, S("a1bcde"));
    test(S("abcde"), 1, 0, str, str+2, S("a12bcde"));

    return true;
}

template <class S>
TEST_CONSTEXPR_CXX20 bool test1()
{
    test(S("abcde"), 1, 0, str, str+4, S("a1234bcde"));
    test(S("abcde"), 1, 0, str, str+5, S("a12345bcde"));
    test(S("abcde"), 1, 0, str, str+0, S("abcde"));
    test(S("abcde"), 1, 0, str, str+1, S("a1bcde"));
    test(S("abcde"), 1, 0, str, str+5, S("a12345bcde"));
    test(S("abcde"), 1, 0, str, str+9, S("a123456789bcde"));
    test(S("abcde"), 1, 0, str, str+10, S("a1234567890bcde"));
    test(S("abcde"), 1, 0, str, str+0, S("abcde"));
    test(S("abcde"), 1, 0, str, str+1, S("a1bcde"));
    test(S("abcde"), 1, 0, str, str+10, S("a1234567890bcde"));
    test(S("abcde"), 1, 0, str, str+19, S("a1234567890123456789bcde"));
    test(S("abcde"), 1, 0, str, str+20, S("a12345678901234567890bcde"));
    test(S("abcde"), 1, 1, str, str+0, S("acde"));
    test(S("abcde"), 1, 1, str, str+0, S("acde"));
    test(S("abcde"), 1, 1, str, str+1, S("a1cde"));
    test(S("abcde"), 1, 1, str, str+2, S("a12cde"));
    test(S("abcde"), 1, 1, str, str+4, S("a1234cde"));
    test(S("abcde"), 1, 1, str, str+5, S("a12345cde"));
    test(S("abcde"), 1, 1, str, str+0, S("acde"));
    test(S("abcde"), 1, 1, str, str+1, S("a1cde"));
    test(S("abcde"), 1, 1, str, str+5, S("a12345cde"));
    test(S("abcde"), 1, 1, str, str+9, S("a123456789cde"));
    test(S("abcde"), 1, 1, str, str+10, S("a1234567890cde"));
    test(S("abcde"), 1, 1, str, str+0, S("acde"));
    test(S("abcde"), 1, 1, str, str+1, S("a1cde"));
    test(S("abcde"), 1, 1, str, str+10, S("a1234567890cde"));
    test(S("abcde"), 1, 1, str, str+19, S("a1234567890123456789cde"));
    test(S("abcde"), 1, 1, str, str+20, S("a12345678901234567890cde"));
    test(S("abcde"), 1, 2, str, str+0, S("ade"));
    test(S("abcde"), 1, 2, str, str+0, S("ade"));
    test(S("abcde"), 1, 2, str, str+1, S("a1de"));
    test(S("abcde"), 1, 2, str, str+2, S("a12de"));
    test(S("abcde"), 1, 2, str, str+4, S("a1234de"));
    test(S("abcde"), 1, 2, str, str+5, S("a12345de"));
    test(S("abcde"), 1, 2, str, str+0, S("ade"));
    test(S("abcde"), 1, 2, str, str+1, S("a1de"));
    test(S("abcde"), 1, 2, str, str+5, S("a12345de"));
    test(S("abcde"), 1, 2, str, str+9, S("a123456789de"));
    test(S("abcde"), 1, 2, str, str+10, S("a1234567890de"));
    test(S("abcde"), 1, 2, str, str+0, S("ade"));
    test(S("abcde"), 1, 2, str, str+1, S("a1de"));
    test(S("abcde"), 1, 2, str, str+10, S("a1234567890de"));
    test(S("abcde"), 1, 2, str, str+19, S("a1234567890123456789de"));
    test(S("abcde"), 1, 2, str, str+20, S("a12345678901234567890de"));
    test(S("abcde"), 1, 3, str, str+0, S("ae"));
    test(S("abcde"), 1, 3, str, str+0, S("ae"));
    test(S("abcde"), 1, 3, str, str+1, S("a1e"));
    test(S("abcde"), 1, 3, str, str+2, S("a12e"));
    test(S("abcde"), 1, 3, str, str+4, S("a1234e"));
    test(S("abcde"), 1, 3, str, str+5, S("a12345e"));
    test(S("abcde"), 1, 3, str, str+0, S("ae"));
    test(S("abcde"), 1, 3, str, str+1, S("a1e"));
    test(S("abcde"), 1, 3, str, str+5, S("a12345e"));
    test(S("abcde"), 1, 3, str, str+9, S("a123456789e"));
    test(S("abcde"), 1, 3, str, str+10, S("a1234567890e"));
    test(S("abcde"), 1, 3, str, str+0, S("ae"));
    test(S("abcde"), 1, 3, str, str+1, S("a1e"));
    test(S("abcde"), 1, 3, str, str+10, S("a1234567890e"));
    test(S("abcde"), 1, 3, str, str+19, S("a1234567890123456789e"));
    test(S("abcde"), 1, 3, str, str+20, S("a12345678901234567890e"));
    test(S("abcde"), 1, 4, str, str+0, S("a"));
    test(S("abcde"), 1, 4, str, str+0, S("a"));
    test(S("abcde"), 1, 4, str, str+1, S("a1"));
    test(S("abcde"), 1, 4, str, str+2, S("a12"));
    test(S("abcde"), 1, 4, str, str+4, S("a1234"));
    test(S("abcde"), 1, 4, str, str+5, S("a12345"));
    test(S("abcde"), 1, 4, str, str+0, S("a"));
    test(S("abcde"), 1, 4, str, str+1, S("a1"));
    test(S("abcde"), 1, 4, str, str+5, S("a12345"));
    test(S("abcde"), 1, 4, str, str+9, S("a123456789"));
    test(S("abcde"), 1, 4, str, str+10, S("a1234567890"));
    test(S("abcde"), 1, 4, str, str+0, S("a"));
    test(S("abcde"), 1, 4, str, str+1, S("a1"));
    test(S("abcde"), 1, 4, str, str+10, S("a1234567890"));
    test(S("abcde"), 1, 4, str, str+19, S("a1234567890123456789"));
    test(S("abcde"), 1, 4, str, str+20, S("a12345678901234567890"));
    test(S("abcde"), 2, 0, str, str+0, S("abcde"));
    test(S("abcde"), 2, 0, str, str+0, S("abcde"));
    test(S("abcde"), 2, 0, str, str+1, S("ab1cde"));
    test(S("abcde"), 2, 0, str, str+2, S("ab12cde"));
    test(S("abcde"), 2, 0, str, str+4, S("ab1234cde"));
    test(S("abcde"), 2, 0, str, str+5, S("ab12345cde"));
    test(S("abcde"), 2, 0, str, str+0, S("abcde"));
    test(S("abcde"), 2, 0, str, str+1, S("ab1cde"));
    test(S("abcde"), 2, 0, str, str+5, S("ab12345cde"));
    test(S("abcde"), 2, 0, str, str+9, S("ab123456789cde"));
    test(S("abcde"), 2, 0, str, str+10, S("ab1234567890cde"));
    test(S("abcde"), 2, 0, str, str+0, S("abcde"));
    test(S("abcde"), 2, 0, str, str+1, S("ab1cde"));
    test(S("abcde"), 2, 0, str, str+10, S("ab1234567890cde"));
    test(S("abcde"), 2, 0, str, str+19, S("ab1234567890123456789cde"));
    test(S("abcde"), 2, 0, str, str+20, S("ab12345678901234567890cde"));
    test(S("abcde"), 2, 1, str, str+0, S("abde"));
    test(S("abcde"), 2, 1, str, str+0, S("abde"));
    test(S("abcde"), 2, 1, str, str+1, S("ab1de"));
    test(S("abcde"), 2, 1, str, str+2, S("ab12de"));
    test(S("abcde"), 2, 1, str, str+4, S("ab1234de"));
    test(S("abcde"), 2, 1, str, str+5, S("ab12345de"));
    test(S("abcde"), 2, 1, str, str+0, S("abde"));
    test(S("abcde"), 2, 1, str, str+1, S("ab1de"));

    return true;
}

template <class S>
TEST_CONSTEXPR_CXX20 bool test2()
{
    test(S("abcde"), 2, 1, str, str+5, S("ab12345de"));
    test(S("abcde"), 2, 1, str, str+9, S("ab123456789de"));
    test(S("abcde"), 2, 1, str, str+10, S("ab1234567890de"));
    test(S("abcde"), 2, 1, str, str+0, S("abde"));
    test(S("abcde"), 2, 1, str, str+1, S("ab1de"));
    test(S("abcde"), 2, 1, str, str+10, S("ab1234567890de"));
    test(S("abcde"), 2, 1, str, str+19, S("ab1234567890123456789de"));
    test(S("abcde"), 2, 1, str, str+20, S("ab12345678901234567890de"));
    test(S("abcde"), 2, 2, str, str+0, S("abe"));
    test(S("abcde"), 2, 2, str, str+0, S("abe"));
    test(S("abcde"), 2, 2, str, str+1, S("ab1e"));
    test(S("abcde"), 2, 2, str, str+2, S("ab12e"));
    test(S("abcde"), 2, 2, str, str+4, S("ab1234e"));
    test(S("abcde"), 2, 2, str, str+5, S("ab12345e"));
    test(S("abcde"), 2, 2, str, str+0, S("abe"));
    test(S("abcde"), 2, 2, str, str+1, S("ab1e"));
    test(S("abcde"), 2, 2, str, str+5, S("ab12345e"));
    test(S("abcde"), 2, 2, str, str+9, S("ab123456789e"));
    test(S("abcde"), 2, 2, str, str+10, S("ab1234567890e"));
    test(S("abcde"), 2, 2, str, str+0, S("abe"));
    test(S("abcde"), 2, 2, str, str+1, S("ab1e"));
    test(S("abcde"), 2, 2, str, str+10, S("ab1234567890e"));
    test(S("abcde"), 2, 2, str, str+19, S("ab1234567890123456789e"));
    test(S("abcde"), 2, 2, str, str+20, S("ab12345678901234567890e"));
    test(S("abcde"), 2, 3, str, str+0, S("ab"));
    test(S("abcde"), 2, 3, str, str+0, S("ab"));
    test(S("abcde"), 2, 3, str, str+1, S("ab1"));
    test(S("abcde"), 2, 3, str, str+2, S("ab12"));
    test(S("abcde"), 2, 3, str, str+4, S("ab1234"));
    test(S("abcde"), 2, 3, str, str+5, S("ab12345"));
    test(S("abcde"), 2, 3, str, str+0, S("ab"));
    test(S("abcde"), 2, 3, str, str+1, S("ab1"));
    test(S("abcde"), 2, 3, str, str+5, S("ab12345"));
    test(S("abcde"), 2, 3, str, str+9, S("ab123456789"));
    test(S("abcde"), 2, 3, str, str+10, S("ab1234567890"));
    test(S("abcde"), 2, 3, str, str+0, S("ab"));
    test(S("abcde"), 2, 3, str, str+1, S("ab1"));
    test(S("abcde"), 2, 3, str, str+10, S("ab1234567890"));
    test(S("abcde"), 2, 3, str, str+19, S("ab1234567890123456789"));
    test(S("abcde"), 2, 3, str, str+20, S("ab12345678901234567890"));
    test(S("abcde"), 4, 0, str, str+0, S("abcde"));
    test(S("abcde"), 4, 0, str, str+0, S("abcde"));
    test(S("abcde"), 4, 0, str, str+1, S("abcd1e"));
    test(S("abcde"), 4, 0, str, str+2, S("abcd12e"));
    test(S("abcde"), 4, 0, str, str+4, S("abcd1234e"));
    test(S("abcde"), 4, 0, str, str+5, S("abcd12345e"));
    test(S("abcde"), 4, 0, str, str+0, S("abcde"));
    test(S("abcde"), 4, 0, str, str+1, S("abcd1e"));
    test(S("abcde"), 4, 0, str, str+5, S("abcd12345e"));
    test(S("abcde"), 4, 0, str, str+9, S("abcd123456789e"));
    test(S("abcde"), 4, 0, str, str+10, S("abcd1234567890e"));
    test(S("abcde"), 4, 0, str, str+0, S("abcde"));
    test(S("abcde"), 4, 0, str, str+1, S("abcd1e"));
    test(S("abcde"), 4, 0, str, str+10, S("abcd1234567890e"));
    test(S("abcde"), 4, 0, str, str+19, S("abcd1234567890123456789e"));
    test(S("abcde"), 4, 0, str, str+20, S("abcd12345678901234567890e"));
    test(S("abcde"), 4, 1, str, str+0, S("abcd"));
    test(S("abcde"), 4, 1, str, str+0, S("abcd"));
    test(S("abcde"), 4, 1, str, str+1, S("abcd1"));
    test(S("abcde"), 4, 1, str, str+2, S("abcd12"));
    test(S("abcde"), 4, 1, str, str+4, S("abcd1234"));
    test(S("abcde"), 4, 1, str, str+5, S("abcd12345"));
    test(S("abcde"), 4, 1, str, str+0, S("abcd"));
    test(S("abcde"), 4, 1, str, str+1, S("abcd1"));
    test(S("abcde"), 4, 1, str, str+5, S("abcd12345"));
    test(S("abcde"), 4, 1, str, str+9, S("abcd123456789"));
    test(S("abcde"), 4, 1, str, str+10, S("abcd1234567890"));
    test(S("abcde"), 4, 1, str, str+0, S("abcd"));
    test(S("abcde"), 4, 1, str, str+1, S("abcd1"));
    test(S("abcde"), 4, 1, str, str+10, S("abcd1234567890"));
    test(S("abcde"), 4, 1, str, str+19, S("abcd1234567890123456789"));
    test(S("abcde"), 4, 1, str, str+20, S("abcd12345678901234567890"));
    test(S("abcde"), 5, 0, str, str+0, S("abcde"));
    test(S("abcde"), 5, 0, str, str+0, S("abcde"));
    test(S("abcde"), 5, 0, str, str+1, S("abcde1"));
    test(S("abcde"), 5, 0, str, str+2, S("abcde12"));
    test(S("abcde"), 5, 0, str, str+4, S("abcde1234"));
    test(S("abcde"), 5, 0, str, str+5, S("abcde12345"));
    test(S("abcde"), 5, 0, str, str+0, S("abcde"));
    test(S("abcde"), 5, 0, str, str+1, S("abcde1"));
    test(S("abcde"), 5, 0, str, str+5, S("abcde12345"));
    test(S("abcde"), 5, 0, str, str+9, S("abcde123456789"));
    test(S("abcde"), 5, 0, str, str+10, S("abcde1234567890"));
    test(S("abcde"), 5, 0, str, str+0, S("abcde"));
    test(S("abcde"), 5, 0, str, str+1, S("abcde1"));
    test(S("abcde"), 5, 0, str, str+10, S("abcde1234567890"));
    test(S("abcde"), 5, 0, str, str+19, S("abcde1234567890123456789"));
    test(S("abcde"), 5, 0, str, str+20, S("abcde12345678901234567890"));
    test(S("abcdefghij"), 0, 0, str, str+0, S("abcdefghij"));
    test(S("abcdefghij"), 0, 0, str, str+0, S("abcdefghij"));
    test(S("abcdefghij"), 0, 0, str, str+1, S("1abcdefghij"));
    test(S("abcdefghij"), 0, 0, str, str+2, S("12abcdefghij"));
    test(S("abcdefghij"), 0, 0, str, str+4, S("1234abcdefghij"));
    test(S("abcdefghij"), 0, 0, str, str+5, S("12345abcdefghij"));
    test(S("abcdefghij"), 0, 0, str, str+0, S("abcdefghij"));
    test(S("abcdefghij"), 0, 0, str, str+1, S("1abcdefghij"));
    test(S("abcdefghij"), 0, 0, str, str+5, S("12345abcdefghij"));
    test(S("abcdefghij"), 0, 0, str, str+9, S("123456789abcdefghij"));
    test(S("abcdefghij"), 0, 0, str, str+10, S("1234567890abcdefghij"));
    test(S("abcdefghij"), 0, 0, str, str+0, S("abcdefghij"));

    return true;
}

template <class S>
TEST_CONSTEXPR_CXX20 bool test3()
{
    test(S("abcdefghij"), 0, 0, str, str+1, S("1abcdefghij"));
    test(S("abcdefghij"), 0, 0, str, str+10, S("1234567890abcdefghij"));
    test(S("abcdefghij"), 0, 0, str, str+19, S("1234567890123456789abcdefghij"));
    test(S("abcdefghij"), 0, 0, str, str+20, S("12345678901234567890abcdefghij"));
    test(S("abcdefghij"), 0, 1, str, str+0, S("bcdefghij"));
    test(S("abcdefghij"), 0, 1, str, str+0, S("bcdefghij"));
    test(S("abcdefghij"), 0, 1, str, str+1, S("1bcdefghij"));
    test(S("abcdefghij"), 0, 1, str, str+2, S("12bcdefghij"));
    test(S("abcdefghij"), 0, 1, str, str+4, S("1234bcdefghij"));
    test(S("abcdefghij"), 0, 1, str, str+5, S("12345bcdefghij"));
    test(S("abcdefghij"), 0, 1, str, str+0, S("bcdefghij"));
    test(S("abcdefghij"), 0, 1, str, str+1, S("1bcdefghij"));
    test(S("abcdefghij"), 0, 1, str, str+5, S("12345bcdefghij"));
    test(S("abcdefghij"), 0, 1, str, str+9, S("123456789bcdefghij"));
    test(S("abcdefghij"), 0, 1, str, str+10, S("1234567890bcdefghij"));
    test(S("abcdefghij"), 0, 1, str, str+0, S("bcdefghij"));
    test(S("abcdefghij"), 0, 1, str, str+1, S("1bcdefghij"));
    test(S("abcdefghij"), 0, 1, str, str+10, S("1234567890bcdefghij"));
    test(S("abcdefghij"), 0, 1, str, str+19, S("1234567890123456789bcdefghij"));
    test(S("abcdefghij"), 0, 1, str, str+20, S("12345678901234567890bcdefghij"));
    test(S("abcdefghij"), 0, 5, str, str+0, S("fghij"));
    test(S("abcdefghij"), 0, 5, str, str+0, S("fghij"));
    test(S("abcdefghij"), 0, 5, str, str+1, S("1fghij"));
    test(S("abcdefghij"), 0, 5, str, str+2, S("12fghij"));
    test(S("abcdefghij"), 0, 5, str, str+4, S("1234fghij"));
    test(S("abcdefghij"), 0, 5, str, str+5, S("12345fghij"));
    test(S("abcdefghij"), 0, 5, str, str+0, S("fghij"));
    test(S("abcdefghij"), 0, 5, str, str+1, S("1fghij"));
    test(S("abcdefghij"), 0, 5, str, str+5, S("12345fghij"));
    test(S("abcdefghij"), 0, 5, str, str+9, S("123456789fghij"));
    test(S("abcdefghij"), 0, 5, str, str+10, S("1234567890fghij"));
    test(S("abcdefghij"), 0, 5, str, str+0, S("fghij"));
    test(S("abcdefghij"), 0, 5, str, str+1, S("1fghij"));
    test(S("abcdefghij"), 0, 5, str, str+10, S("1234567890fghij"));
    test(S("abcdefghij"), 0, 5, str, str+19, S("1234567890123456789fghij"));
    test(S("abcdefghij"), 0, 5, str, str+20, S("12345678901234567890fghij"));
    test(S("abcdefghij"), 0, 9, str, str+0, S("j"));
    test(S("abcdefghij"), 0, 9, str, str+0, S("j"));
    test(S("abcdefghij"), 0, 9, str, str+1, S("1j"));
    test(S("abcdefghij"), 0, 9, str, str+2, S("12j"));
    test(S("abcdefghij"), 0, 9, str, str+4, S("1234j"));
    test(S("abcdefghij"), 0, 9, str, str+5, S("12345j"));
    test(S("abcdefghij"), 0, 9, str, str+0, S("j"));
    test(S("abcdefghij"), 0, 9, str, str+1, S("1j"));
    test(S("abcdefghij"), 0, 9, str, str+5, S("12345j"));
    test(S("abcdefghij"), 0, 9, str, str+9, S("123456789j"));
    test(S("abcdefghij"), 0, 9, str, str+10, S("1234567890j"));
    test(S("abcdefghij"), 0, 9, str, str+0, S("j"));
    test(S("abcdefghij"), 0, 9, str, str+1, S("1j"));
    test(S("abcdefghij"), 0, 9, str, str+10, S("1234567890j"));
    test(S("abcdefghij"), 0, 9, str, str+19, S("1234567890123456789j"));
    test(S("abcdefghij"), 0, 9, str, str+20, S("12345678901234567890j"));
    test(S("abcdefghij"), 0, 10, str, str+0, S(""));
    test(S("abcdefghij"), 0, 10, str, str+0, S(""));
    test(S("abcdefghij"), 0, 10, str, str+1, S("1"));
    test(S("abcdefghij"), 0, 10, str, str+2, S("12"));
    test(S("abcdefghij"), 0, 10, str, str+4, S("1234"));
    test(S("abcdefghij"), 0, 10, str, str+5, S("12345"));
    test(S("abcdefghij"), 0, 10, str, str+0, S(""));
    test(S("abcdefghij"), 0, 10, str, str+1, S("1"));
    test(S("abcdefghij"), 0, 10, str, str+5, S("12345"));
    test(S("abcdefghij"), 0, 10, str, str+9, S("123456789"));
    test(S("abcdefghij"), 0, 10, str, str+10, S("1234567890"));
    test(S("abcdefghij"), 0, 10, str, str+0, S(""));
    test(S("abcdefghij"), 0, 10, str, str+1, S("1"));
    test(S("abcdefghij"), 0, 10, str, str+10, S("1234567890"));
    test(S("abcdefghij"), 0, 10, str, str+19, S("1234567890123456789"));
    test(S("abcdefghij"), 0, 10, str, str+20, S("12345678901234567890"));
    test(S("abcdefghij"), 1, 0, str, str+0, S("abcdefghij"));
    test(S("abcdefghij"), 1, 0, str, str+0, S("abcdefghij"));
    test(S("abcdefghij"), 1, 0, str, str+1, S("a1bcdefghij"));
    test(S("abcdefghij"), 1, 0, str, str+2, S("a12bcdefghij"));
    test(S("abcdefghij"), 1, 0, str, str+4, S("a1234bcdefghij"));
    test(S("abcdefghij"), 1, 0, str, str+5, S("a12345bcdefghij"));
    test(S("abcdefghij"), 1, 0, str, str+0, S("abcdefghij"));
    test(S("abcdefghij"), 1, 0, str, str+1, S("a1bcdefghij"));
    test(S("abcdefghij"), 1, 0, str, str+5, S("a12345bcdefghij"));
    test(S("abcdefghij"), 1, 0, str, str+9, S("a123456789bcdefghij"));
    test(S("abcdefghij"), 1, 0, str, str+10, S("a1234567890bcdefghij"));
    test(S("abcdefghij"), 1, 0, str, str+0, S("abcdefghij"));
    test(S("abcdefghij"), 1, 0, str, str+1, S("a1bcdefghij"));
    test(S("abcdefghij"), 1, 0, str, str+10, S("a1234567890bcdefghij"));
    test(S("abcdefghij"), 1, 0, str, str+19, S("a1234567890123456789bcdefghij"));
    test(S("abcdefghij"), 1, 0, str, str+20, S("a12345678901234567890bcdefghij"));
    test(S("abcdefghij"), 1, 1, str, str+0, S("acdefghij"));
    test(S("abcdefghij"), 1, 1, str, str+0, S("acdefghij"));
    test(S("abcdefghij"), 1, 1, str, str+1, S("a1cdefghij"));
    test(S("abcdefghij"), 1, 1, str, str+2, S("a12cdefghij"));
    test(S("abcdefghij"), 1, 1, str, str+4, S("a1234cdefghij"));
    test(S("abcdefghij"), 1, 1, str, str+5, S("a12345cdefghij"));
    test(S("abcdefghij"), 1, 1, str, str+0, S("acdefghij"));
    test(S("abcdefghij"), 1, 1, str, str+1, S("a1cdefghij"));
    test(S("abcdefghij"), 1, 1, str, str+5, S("a12345cdefghij"));
    test(S("abcdefghij"), 1, 1, str, str+9, S("a123456789cdefghij"));
    test(S("abcdefghij"), 1, 1, str, str+10, S("a1234567890cdefghij"));
    test(S("abcdefghij"), 1, 1, str, str+0, S("acdefghij"));
    test(S("abcdefghij"), 1, 1, str, str+1, S("a1cdefghij"));
    test(S("abcdefghij"), 1, 1, str, str+10, S("a1234567890cdefghij"));
    test(S("abcdefghij"), 1, 1, str, str+19, S("a1234567890123456789cdefghij"));
    test(S("abcdefghij"), 1, 1, str, str+20, S("a12345678901234567890cdefghij"));

    return true;
}

template <class S>
TEST_CONSTEXPR_CXX20 bool test4()
{
    test(S("abcdefghij"), 1, 4, str, str+0, S("afghij"));
    test(S("abcdefghij"), 1, 4, str, str+0, S("afghij"));
    test(S("abcdefghij"), 1, 4, str, str+1, S("a1fghij"));
    test(S("abcdefghij"), 1, 4, str, str+2, S("a12fghij"));
    test(S("abcdefghij"), 1, 4, str, str+4, S("a1234fghij"));
    test(S("abcdefghij"), 1, 4, str, str+5, S("a12345fghij"));
    test(S("abcdefghij"), 1, 4, str, str+0, S("afghij"));
    test(S("abcdefghij"), 1, 4, str, str+1, S("a1fghij"));
    test(S("abcdefghij"), 1, 4, str, str+5, S("a12345fghij"));
    test(S("abcdefghij"), 1, 4, str, str+9, S("a123456789fghij"));
    test(S("abcdefghij"), 1, 4, str, str+10, S("a1234567890fghij"));
    test(S("abcdefghij"), 1, 4, str, str+0, S("afghij"));
    test(S("abcdefghij"), 1, 4, str, str+1, S("a1fghij"));
    test(S("abcdefghij"), 1, 4, str, str+10, S("a1234567890fghij"));
    test(S("abcdefghij"), 1, 4, str, str+19, S("a1234567890123456789fghij"));
    test(S("abcdefghij"), 1, 4, str, str+20, S("a12345678901234567890fghij"));
    test(S("abcdefghij"), 1, 8, str, str+0, S("aj"));
    test(S("abcdefghij"), 1, 8, str, str+0, S("aj"));
    test(S("abcdefghij"), 1, 8, str, str+1, S("a1j"));
    test(S("abcdefghij"), 1, 8, str, str+2, S("a12j"));
    test(S("abcdefghij"), 1, 8, str, str+4, S("a1234j"));
    test(S("abcdefghij"), 1, 8, str, str+5, S("a12345j"));
    test(S("abcdefghij"), 1, 8, str, str+0, S("aj"));
    test(S("abcdefghij"), 1, 8, str, str+1, S("a1j"));
    test(S("abcdefghij"), 1, 8, str, str+5, S("a12345j"));
    test(S("abcdefghij"), 1, 8, str, str+9, S("a123456789j"));
    test(S("abcdefghij"), 1, 8, str, str+10, S("a1234567890j"));
    test(S("abcdefghij"), 1, 8, str, str+0, S("aj"));
    test(S("abcdefghij"), 1, 8, str, str+1, S("a1j"));
    test(S("abcdefghij"), 1, 8, str, str+10, S("a1234567890j"));
    test(S("abcdefghij"), 1, 8, str, str+19, S("a1234567890123456789j"));
    test(S("abcdefghij"), 1, 8, str, str+20, S("a12345678901234567890j"));
    test(S("abcdefghij"), 1, 9, str, str+0, S("a"));
    test(S("abcdefghij"), 1, 9, str, str+0, S("a"));
    test(S("abcdefghij"), 1, 9, str, str+1, S("a1"));
    test(S("abcdefghij"), 1, 9, str, str+2, S("a12"));
    test(S("abcdefghij"), 1, 9, str, str+4, S("a1234"));
    test(S("abcdefghij"), 1, 9, str, str+5, S("a12345"));
    test(S("abcdefghij"), 1, 9, str, str+0, S("a"));
    test(S("abcdefghij"), 1, 9, str, str+1, S("a1"));
    test(S("abcdefghij"), 1, 9, str, str+5, S("a12345"));
    test(S("abcdefghij"), 1, 9, str, str+9, S("a123456789"));
    test(S("abcdefghij"), 1, 9, str, str+10, S("a1234567890"));
    test(S("abcdefghij"), 1, 9, str, str+0, S("a"));
    test(S("abcdefghij"), 1, 9, str, str+1, S("a1"));
    test(S("abcdefghij"), 1, 9, str, str+10, S("a1234567890"));
    test(S("abcdefghij"), 1, 9, str, str+19, S("a1234567890123456789"));
    test(S("abcdefghij"), 1, 9, str, str+20, S("a12345678901234567890"));
    test(S("abcdefghij"), 5, 0, str, str+0, S("abcdefghij"));
    test(S("abcdefghij"), 5, 0, str, str+0, S("abcdefghij"));
    test(S("abcdefghij"), 5, 0, str, str+1, S("abcde1fghij"));
    test(S("abcdefghij"), 5, 0, str, str+2, S("abcde12fghij"));
    test(S("abcdefghij"), 5, 0, str, str+4, S("abcde1234fghij"));
    test(S("abcdefghij"), 5, 0, str, str+5, S("abcde12345fghij"));
    test(S("abcdefghij"), 5, 0, str, str+0, S("abcdefghij"));
    test(S("abcdefghij"), 5, 0, str, str+1, S("abcde1fghij"));
    test(S("abcdefghij"), 5, 0, str, str+5, S("abcde12345fghij"));
    test(S("abcdefghij"), 5, 0, str, str+9, S("abcde123456789fghij"));
    test(S("abcdefghij"), 5, 0, str, str+10, S("abcde1234567890fghij"));
    test(S("abcdefghij"), 5, 0, str, str+0, S("abcdefghij"));
    test(S("abcdefghij"), 5, 0, str, str+1, S("abcde1fghij"));
    test(S("abcdefghij"), 5, 0, str, str+10, S("abcde1234567890fghij"));
    test(S("abcdefghij"), 5, 0, str, str+19, S("abcde1234567890123456789fghij"));
    test(S("abcdefghij"), 5, 0, str, str+20, S("abcde12345678901234567890fghij"));
    test(S("abcdefghij"), 5, 1, str, str+0, S("abcdeghij"));
    test(S("abcdefghij"), 5, 1, str, str+0, S("abcdeghij"));
    test(S("abcdefghij"), 5, 1, str, str+1, S("abcde1ghij"));
    test(S("abcdefghij"), 5, 1, str, str+2, S("abcde12ghij"));
    test(S("abcdefghij"), 5, 1, str, str+4, S("abcde1234ghij"));
    test(S("abcdefghij"), 5, 1, str, str+5, S("abcde12345ghij"));
    test(S("abcdefghij"), 5, 1, str, str+0, S("abcdeghij"));
    test(S("abcdefghij"), 5, 1, str, str+1, S("abcde1ghij"));
    test(S("abcdefghij"), 5, 1, str, str+5, S("abcde12345ghij"));
    test(S("abcdefghij"), 5, 1, str, str+9, S("abcde123456789ghij"));
    test(S("abcdefghij"), 5, 1, str, str+10, S("abcde1234567890ghij"));
    test(S("abcdefghij"), 5, 1, str, str+0, S("abcdeghij"));
    test(S("abcdefghij"), 5, 1, str, str+1, S("abcde1ghij"));
    test(S("abcdefghij"), 5, 1, str, str+10, S("abcde1234567890ghij"));
    test(S("abcdefghij"), 5, 1, str, str+19, S("abcde1234567890123456789ghij"));
    test(S("abcdefghij"), 5, 1, str, str+20, S("abcde12345678901234567890ghij"));
    test(S("abcdefghij"), 5, 2, str, str+0, S("abcdehij"));
    test(S("abcdefghij"), 5, 2, str, str+0, S("abcdehij"));
    test(S("abcdefghij"), 5, 2, str, str+1, S("abcde1hij"));
    test(S("abcdefghij"), 5, 2, str, str+2, S("abcde12hij"));
    test(S("abcdefghij"), 5, 2, str, str+4, S("abcde1234hij"));
    test(S("abcdefghij"), 5, 2, str, str+5, S("abcde12345hij"));
    test(S("abcdefghij"), 5, 2, str, str+0, S("abcdehij"));
    test(S("abcdefghij"), 5, 2, str, str+1, S("abcde1hij"));
    test(S("abcdefghij"), 5, 2, str, str+5, S("abcde12345hij"));
    test(S("abcdefghij"), 5, 2, str, str+9, S("abcde123456789hij"));
    test(S("abcdefghij"), 5, 2, str, str+10, S("abcde1234567890hij"));
    test(S("abcdefghij"), 5, 2, str, str+0, S("abcdehij"));
    test(S("abcdefghij"), 5, 2, str, str+1, S("abcde1hij"));
    test(S("abcdefghij"), 5, 2, str, str+10, S("abcde1234567890hij"));
    test(S("abcdefghij"), 5, 2, str, str+19, S("abcde1234567890123456789hij"));
    test(S("abcdefghij"), 5, 2, str, str+20, S("abcde12345678901234567890hij"));
    test(S("abcdefghij"), 5, 4, str, str+0, S("abcdej"));
    test(S("abcdefghij"), 5, 4, str, str+0, S("abcdej"));
    test(S("abcdefghij"), 5, 4, str, str+1, S("abcde1j"));
    test(S("abcdefghij"), 5, 4, str, str+2, S("abcde12j"));

    return true;
}

template <class S>
TEST_CONSTEXPR_CXX20 bool test5()
{
    test(S("abcdefghij"), 5, 4, str, str+4, S("abcde1234j"));
    test(S("abcdefghij"), 5, 4, str, str+5, S("abcde12345j"));
    test(S("abcdefghij"), 5, 4, str, str+0, S("abcdej"));
    test(S("abcdefghij"), 5, 4, str, str+1, S("abcde1j"));
    test(S("abcdefghij"), 5, 4, str, str+5, S("abcde12345j"));
    test(S("abcdefghij"), 5, 4, str, str+9, S("abcde123456789j"));
    test(S("abcdefghij"), 5, 4, str, str+10, S("abcde1234567890j"));
    test(S("abcdefghij"), 5, 4, str, str+0, S("abcdej"));
    test(S("abcdefghij"), 5, 4, str, str+1, S("abcde1j"));
    test(S("abcdefghij"), 5, 4, str, str+10, S("abcde1234567890j"));
    test(S("abcdefghij"), 5, 4, str, str+19, S("abcde1234567890123456789j"));
    test(S("abcdefghij"), 5, 4, str, str+20, S("abcde12345678901234567890j"));
    test(S("abcdefghij"), 5, 5, str, str+0, S("abcde"));
    test(S("abcdefghij"), 5, 5, str, str+0, S("abcde"));
    test(S("abcdefghij"), 5, 5, str, str+1, S("abcde1"));
    test(S("abcdefghij"), 5, 5, str, str+2, S("abcde12"));
    test(S("abcdefghij"), 5, 5, str, str+4, S("abcde1234"));
    test(S("abcdefghij"), 5, 5, str, str+5, S("abcde12345"));
    test(S("abcdefghij"), 5, 5, str, str+0, S("abcde"));
    test(S("abcdefghij"), 5, 5, str, str+1, S("abcde1"));
    test(S("abcdefghij"), 5, 5, str, str+5, S("abcde12345"));
    test(S("abcdefghij"), 5, 5, str, str+9, S("abcde123456789"));
    test(S("abcdefghij"), 5, 5, str, str+10, S("abcde1234567890"));
    test(S("abcdefghij"), 5, 5, str, str+0, S("abcde"));
    test(S("abcdefghij"), 5, 5, str, str+1, S("abcde1"));
    test(S("abcdefghij"), 5, 5, str, str+10, S("abcde1234567890"));
    test(S("abcdefghij"), 5, 5, str, str+19, S("abcde1234567890123456789"));
    test(S("abcdefghij"), 5, 5, str, str+20, S("abcde12345678901234567890"));
    test(S("abcdefghij"), 9, 0, str, str+0, S("abcdefghij"));
    test(S("abcdefghij"), 9, 0, str, str+0, S("abcdefghij"));
    test(S("abcdefghij"), 9, 0, str, str+1, S("abcdefghi1j"));
    test(S("abcdefghij"), 9, 0, str, str+2, S("abcdefghi12j"));
    test(S("abcdefghij"), 9, 0, str, str+4, S("abcdefghi1234j"));
    test(S("abcdefghij"), 9, 0, str, str+5, S("abcdefghi12345j"));
    test(S("abcdefghij"), 9, 0, str, str+0, S("abcdefghij"));
    test(S("abcdefghij"), 9, 0, str, str+1, S("abcdefghi1j"));
    test(S("abcdefghij"), 9, 0, str, str+5, S("abcdefghi12345j"));
    test(S("abcdefghij"), 9, 0, str, str+9, S("abcdefghi123456789j"));
    test(S("abcdefghij"), 9, 0, str, str+10, S("abcdefghi1234567890j"));
    test(S("abcdefghij"), 9, 0, str, str+0, S("abcdefghij"));
    test(S("abcdefghij"), 9, 0, str, str+1, S("abcdefghi1j"));
    test(S("abcdefghij"), 9, 0, str, str+10, S("abcdefghi1234567890j"));
    test(S("abcdefghij"), 9, 0, str, str+19, S("abcdefghi1234567890123456789j"));
    test(S("abcdefghij"), 9, 0, str, str+20, S("abcdefghi12345678901234567890j"));
    test(S("abcdefghij"), 9, 1, str, str+0, S("abcdefghi"));
    test(S("abcdefghij"), 9, 1, str, str+0, S("abcdefghi"));
    test(S("abcdefghij"), 9, 1, str, str+1, S("abcdefghi1"));
    test(S("abcdefghij"), 9, 1, str, str+2, S("abcdefghi12"));
    test(S("abcdefghij"), 9, 1, str, str+4, S("abcdefghi1234"));
    test(S("abcdefghij"), 9, 1, str, str+5, S("abcdefghi12345"));
    test(S("abcdefghij"), 9, 1, str, str+0, S("abcdefghi"));
    test(S("abcdefghij"), 9, 1, str, str+1, S("abcdefghi1"));
    test(S("abcdefghij"), 9, 1, str, str+5, S("abcdefghi12345"));
    test(S("abcdefghij"), 9, 1, str, str+9, S("abcdefghi123456789"));
    test(S("abcdefghij"), 9, 1, str, str+10, S("abcdefghi1234567890"));
    test(S("abcdefghij"), 9, 1, str, str+0, S("abcdefghi"));
    test(S("abcdefghij"), 9, 1, str, str+1, S("abcdefghi1"));
    test(S("abcdefghij"), 9, 1, str, str+10, S("abcdefghi1234567890"));
    test(S("abcdefghij"), 9, 1, str, str+19, S("abcdefghi1234567890123456789"));
    test(S("abcdefghij"), 9, 1, str, str+20, S("abcdefghi12345678901234567890"));
    test(S("abcdefghij"), 10, 0, str, str+0, S("abcdefghij"));
    test(S("abcdefghij"), 10, 0, str, str+0, S("abcdefghij"));
    test(S("abcdefghij"), 10, 0, str, str+1, S("abcdefghij1"));
    test(S("abcdefghij"), 10, 0, str, str+2, S("abcdefghij12"));
    test(S("abcdefghij"), 10, 0, str, str+4, S("abcdefghij1234"));
    test(S("abcdefghij"), 10, 0, str, str+5, S("abcdefghij12345"));
    test(S("abcdefghij"), 10, 0, str, str+0, S("abcdefghij"));
    test(S("abcdefghij"), 10, 0, str, str+1, S("abcdefghij1"));
    test(S("abcdefghij"), 10, 0, str, str+5, S("abcdefghij12345"));
    test(S("abcdefghij"), 10, 0, str, str+9, S("abcdefghij123456789"));
    test(S("abcdefghij"), 10, 0, str, str+10, S("abcdefghij1234567890"));
    test(S("abcdefghij"), 10, 0, str, str+0, S("abcdefghij"));
    test(S("abcdefghij"), 10, 0, str, str+1, S("abcdefghij1"));
    test(S("abcdefghij"), 10, 0, str, str+10, S("abcdefghij1234567890"));
    test(S("abcdefghij"), 10, 0, str, str+19, S("abcdefghij1234567890123456789"));
    test(S("abcdefghij"), 10, 0, str, str+20, S("abcdefghij12345678901234567890"));
    test(S("abcdefghijklmnopqrst"), 0, 0, str, str+0, S("abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 0, str, str+0, S("abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 0, str, str+1, S("1abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 0, str, str+2, S("12abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 0, str, str+4, S("1234abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 0, str, str+5, S("12345abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 0, str, str+0, S("abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 0, str, str+1, S("1abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 0, str, str+5, S("12345abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 0, str, str+9, S("123456789abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 0, str, str+10, S("1234567890abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 0, str, str+0, S("abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 0, str, str+1, S("1abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 0, str, str+10, S("1234567890abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 0, str, str+19, S("1234567890123456789abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 0, str, str+20, S("12345678901234567890abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 1, str, str+0, S("bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 1, str, str+0, S("bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 1, str, str+1, S("1bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 1, str, str+2, S("12bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 1, str, str+4, S("1234bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 1, str, str+5, S("12345bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 1, str, str+0, S("bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 1, str, str+1, S("1bcdefghijklmnopqrst"));

    return true;
}

template <class S>
TEST_CONSTEXPR_CXX20 bool test6()
{
    test(S("abcdefghijklmnopqrst"), 0, 1, str, str+5, S("12345bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 1, str, str+9, S("123456789bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 1, str, str+10, S("1234567890bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 1, str, str+0, S("bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 1, str, str+1, S("1bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 1, str, str+10, S("1234567890bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 1, str, str+19, S("1234567890123456789bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 1, str, str+20, S("12345678901234567890bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 10, str, str+0, S("klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 10, str, str+0, S("klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 10, str, str+1, S("1klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 10, str, str+2, S("12klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 10, str, str+4, S("1234klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 10, str, str+5, S("12345klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 10, str, str+0, S("klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 10, str, str+1, S("1klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 10, str, str+5, S("12345klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 10, str, str+9, S("123456789klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 10, str, str+10, S("1234567890klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 10, str, str+0, S("klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 10, str, str+1, S("1klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 10, str, str+10, S("1234567890klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 10, str, str+19, S("1234567890123456789klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 10, str, str+20, S("12345678901234567890klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 0, 19, str, str+0, S("t"));
    test(S("abcdefghijklmnopqrst"), 0, 19, str, str+0, S("t"));
    test(S("abcdefghijklmnopqrst"), 0, 19, str, str+1, S("1t"));
    test(S("abcdefghijklmnopqrst"), 0, 19, str, str+2, S("12t"));
    test(S("abcdefghijklmnopqrst"), 0, 19, str, str+4, S("1234t"));
    test(S("abcdefghijklmnopqrst"), 0, 19, str, str+5, S("12345t"));
    test(S("abcdefghijklmnopqrst"), 0, 19, str, str+0, S("t"));
    test(S("abcdefghijklmnopqrst"), 0, 19, str, str+1, S("1t"));
    test(S("abcdefghijklmnopqrst"), 0, 19, str, str+5, S("12345t"));
    test(S("abcdefghijklmnopqrst"), 0, 19, str, str+9, S("123456789t"));
    test(S("abcdefghijklmnopqrst"), 0, 19, str, str+10, S("1234567890t"));
    test(S("abcdefghijklmnopqrst"), 0, 19, str, str+0, S("t"));
    test(S("abcdefghijklmnopqrst"), 0, 19, str, str+1, S("1t"));
    test(S("abcdefghijklmnopqrst"), 0, 19, str, str+10, S("1234567890t"));
    test(S("abcdefghijklmnopqrst"), 0, 19, str, str+19, S("1234567890123456789t"));
    test(S("abcdefghijklmnopqrst"), 0, 19, str, str+20, S("12345678901234567890t"));
    test(S("abcdefghijklmnopqrst"), 0, 20, str, str+0, S(""));
    test(S("abcdefghijklmnopqrst"), 0, 20, str, str+0, S(""));
    test(S("abcdefghijklmnopqrst"), 0, 20, str, str+1, S("1"));
    test(S("abcdefghijklmnopqrst"), 0, 20, str, str+2, S("12"));
    test(S("abcdefghijklmnopqrst"), 0, 20, str, str+4, S("1234"));
    test(S("abcdefghijklmnopqrst"), 0, 20, str, str+5, S("12345"));
    test(S("abcdefghijklmnopqrst"), 0, 20, str, str+0, S(""));
    test(S("abcdefghijklmnopqrst"), 0, 20, str, str+1, S("1"));
    test(S("abcdefghijklmnopqrst"), 0, 20, str, str+5, S("12345"));
    test(S("abcdefghijklmnopqrst"), 0, 20, str, str+9, S("123456789"));
    test(S("abcdefghijklmnopqrst"), 0, 20, str, str+10, S("1234567890"));
    test(S("abcdefghijklmnopqrst"), 0, 20, str, str+0, S(""));
    test(S("abcdefghijklmnopqrst"), 0, 20, str, str+1, S("1"));
    test(S("abcdefghijklmnopqrst"), 0, 20, str, str+10, S("1234567890"));
    test(S("abcdefghijklmnopqrst"), 0, 20, str, str+19, S("1234567890123456789"));
    test(S("abcdefghijklmnopqrst"), 0, 20, str, str+20, S("12345678901234567890"));
    test(S("abcdefghijklmnopqrst"), 1, 0, str, str+0, S("abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 0, str, str+0, S("abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 0, str, str+1, S("a1bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 0, str, str+2, S("a12bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 0, str, str+4, S("a1234bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 0, str, str+5, S("a12345bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 0, str, str+0, S("abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 0, str, str+1, S("a1bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 0, str, str+5, S("a12345bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 0, str, str+9, S("a123456789bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 0, str, str+10, S("a1234567890bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 0, str, str+0, S("abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 0, str, str+1, S("a1bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 0, str, str+10, S("a1234567890bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 0, str, str+19, S("a1234567890123456789bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 0, str, str+20, S("a12345678901234567890bcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 1, str, str+0, S("acdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 1, str, str+0, S("acdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 1, str, str+1, S("a1cdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 1, str, str+2, S("a12cdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 1, str, str+4, S("a1234cdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 1, str, str+5, S("a12345cdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 1, str, str+0, S("acdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 1, str, str+1, S("a1cdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 1, str, str+5, S("a12345cdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 1, str, str+9, S("a123456789cdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 1, str, str+10, S("a1234567890cdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 1, str, str+0, S("acdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 1, str, str+1, S("a1cdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 1, str, str+10, S("a1234567890cdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 1, str, str+19, S("a1234567890123456789cdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 1, str, str+20, S("a12345678901234567890cdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 9, str, str+0, S("aklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 9, str, str+0, S("aklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 9, str, str+1, S("a1klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 9, str, str+2, S("a12klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 9, str, str+4, S("a1234klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 9, str, str+5, S("a12345klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 9, str, str+0, S("aklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 9, str, str+1, S("a1klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 9, str, str+5, S("a12345klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 9, str, str+9, S("a123456789klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 9, str, str+10, S("a1234567890klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 9, str, str+0, S("aklmnopqrst"));

    return true;
}

template <class S>
TEST_CONSTEXPR_CXX20 bool test7()
{
    test(S("abcdefghijklmnopqrst"), 1, 9, str, str+1, S("a1klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 9, str, str+10, S("a1234567890klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 9, str, str+19, S("a1234567890123456789klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 9, str, str+20, S("a12345678901234567890klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 1, 18, str, str+0, S("at"));
    test(S("abcdefghijklmnopqrst"), 1, 18, str, str+0, S("at"));
    test(S("abcdefghijklmnopqrst"), 1, 18, str, str+1, S("a1t"));
    test(S("abcdefghijklmnopqrst"), 1, 18, str, str+2, S("a12t"));
    test(S("abcdefghijklmnopqrst"), 1, 18, str, str+4, S("a1234t"));
    test(S("abcdefghijklmnopqrst"), 1, 18, str, str+5, S("a12345t"));
    test(S("abcdefghijklmnopqrst"), 1, 18, str, str+0, S("at"));
    test(S("abcdefghijklmnopqrst"), 1, 18, str, str+1, S("a1t"));
    test(S("abcdefghijklmnopqrst"), 1, 18, str, str+5, S("a12345t"));
    test(S("abcdefghijklmnopqrst"), 1, 18, str, str+9, S("a123456789t"));
    test(S("abcdefghijklmnopqrst"), 1, 18, str, str+10, S("a1234567890t"));
    test(S("abcdefghijklmnopqrst"), 1, 18, str, str+0, S("at"));
    test(S("abcdefghijklmnopqrst"), 1, 18, str, str+1, S("a1t"));
    test(S("abcdefghijklmnopqrst"), 1, 18, str, str+10, S("a1234567890t"));
    test(S("abcdefghijklmnopqrst"), 1, 18, str, str+19, S("a1234567890123456789t"));
    test(S("abcdefghijklmnopqrst"), 1, 18, str, str+20, S("a12345678901234567890t"));
    test(S("abcdefghijklmnopqrst"), 1, 19, str, str+0, S("a"));
    test(S("abcdefghijklmnopqrst"), 1, 19, str, str+0, S("a"));
    test(S("abcdefghijklmnopqrst"), 1, 19, str, str+1, S("a1"));
    test(S("abcdefghijklmnopqrst"), 1, 19, str, str+2, S("a12"));
    test(S("abcdefghijklmnopqrst"), 1, 19, str, str+4, S("a1234"));
    test(S("abcdefghijklmnopqrst"), 1, 19, str, str+5, S("a12345"));
    test(S("abcdefghijklmnopqrst"), 1, 19, str, str+0, S("a"));
    test(S("abcdefghijklmnopqrst"), 1, 19, str, str+1, S("a1"));
    test(S("abcdefghijklmnopqrst"), 1, 19, str, str+5, S("a12345"));
    test(S("abcdefghijklmnopqrst"), 1, 19, str, str+9, S("a123456789"));
    test(S("abcdefghijklmnopqrst"), 1, 19, str, str+10, S("a1234567890"));
    test(S("abcdefghijklmnopqrst"), 1, 19, str, str+0, S("a"));
    test(S("abcdefghijklmnopqrst"), 1, 19, str, str+1, S("a1"));
    test(S("abcdefghijklmnopqrst"), 1, 19, str, str+10, S("a1234567890"));
    test(S("abcdefghijklmnopqrst"), 1, 19, str, str+19, S("a1234567890123456789"));
    test(S("abcdefghijklmnopqrst"), 1, 19, str, str+20, S("a12345678901234567890"));
    test(S("abcdefghijklmnopqrst"), 10, 0, str, str+0, S("abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 0, str, str+0, S("abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 0, str, str+1, S("abcdefghij1klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 0, str, str+2, S("abcdefghij12klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 0, str, str+4, S("abcdefghij1234klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 0, str, str+5, S("abcdefghij12345klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 0, str, str+0, S("abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 0, str, str+1, S("abcdefghij1klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 0, str, str+5, S("abcdefghij12345klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 0, str, str+9, S("abcdefghij123456789klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 0, str, str+10, S("abcdefghij1234567890klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 0, str, str+0, S("abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 0, str, str+1, S("abcdefghij1klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 0, str, str+10, S("abcdefghij1234567890klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 0, str, str+19, S("abcdefghij1234567890123456789klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 0, str, str+20, S("abcdefghij12345678901234567890klmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 1, str, str+0, S("abcdefghijlmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 1, str, str+0, S("abcdefghijlmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 1, str, str+1, S("abcdefghij1lmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 1, str, str+2, S("abcdefghij12lmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 1, str, str+4, S("abcdefghij1234lmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 1, str, str+5, S("abcdefghij12345lmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 1, str, str+0, S("abcdefghijlmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 1, str, str+1, S("abcdefghij1lmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 1, str, str+5, S("abcdefghij12345lmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 1, str, str+9, S("abcdefghij123456789lmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 1, str, str+10, S("abcdefghij1234567890lmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 1, str, str+0, S("abcdefghijlmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 1, str, str+1, S("abcdefghij1lmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 1, str, str+10, S("abcdefghij1234567890lmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 1, str, str+19, S("abcdefghij1234567890123456789lmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 1, str, str+20, S("abcdefghij12345678901234567890lmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 5, str, str+0, S("abcdefghijpqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 5, str, str+0, S("abcdefghijpqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 5, str, str+1, S("abcdefghij1pqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 5, str, str+2, S("abcdefghij12pqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 5, str, str+4, S("abcdefghij1234pqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 5, str, str+5, S("abcdefghij12345pqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 5, str, str+0, S("abcdefghijpqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 5, str, str+1, S("abcdefghij1pqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 5, str, str+5, S("abcdefghij12345pqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 5, str, str+9, S("abcdefghij123456789pqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 5, str, str+10, S("abcdefghij1234567890pqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 5, str, str+0, S("abcdefghijpqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 5, str, str+1, S("abcdefghij1pqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 5, str, str+10, S("abcdefghij1234567890pqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 5, str, str+19, S("abcdefghij1234567890123456789pqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 5, str, str+20, S("abcdefghij12345678901234567890pqrst"));
    test(S("abcdefghijklmnopqrst"), 10, 9, str, str+0, S("abcdefghijt"));
    test(S("abcdefghijklmnopqrst"), 10, 9, str, str+0, S("abcdefghijt"));
    test(S("abcdefghijklmnopqrst"), 10, 9, str, str+1, S("abcdefghij1t"));
    test(S("abcdefghijklmnopqrst"), 10, 9, str, str+2, S("abcdefghij12t"));
    test(S("abcdefghijklmnopqrst"), 10, 9, str, str+4, S("abcdefghij1234t"));
    test(S("abcdefghijklmnopqrst"), 10, 9, str, str+5, S("abcdefghij12345t"));
    test(S("abcdefghijklmnopqrst"), 10, 9, str, str+0, S("abcdefghijt"));
    test(S("abcdefghijklmnopqrst"), 10, 9, str, str+1, S("abcdefghij1t"));
    test(S("abcdefghijklmnopqrst"), 10, 9, str, str+5, S("abcdefghij12345t"));
    test(S("abcdefghijklmnopqrst"), 10, 9, str, str+9, S("abcdefghij123456789t"));
    test(S("abcdefghijklmnopqrst"), 10, 9, str, str+10, S("abcdefghij1234567890t"));
    test(S("abcdefghijklmnopqrst"), 10, 9, str, str+0, S("abcdefghijt"));
    test(S("abcdefghijklmnopqrst"), 10, 9, str, str+1, S("abcdefghij1t"));
    test(S("abcdefghijklmnopqrst"), 10, 9, str, str+10, S("abcdefghij1234567890t"));
    test(S("abcdefghijklmnopqrst"), 10, 9, str, str+19, S("abcdefghij1234567890123456789t"));
    test(S("abcdefghijklmnopqrst"), 10, 9, str, str+20, S("abcdefghij12345678901234567890t"));

    return true;
}

template <class S>
TEST_CONSTEXPR_CXX20 bool test8()
{
    test(S("abcdefghijklmnopqrst"), 10, 10, str, str+0, S("abcdefghij"));
    test(S("abcdefghijklmnopqrst"), 10, 10, str, str+0, S("abcdefghij"));
    test(S("abcdefghijklmnopqrst"), 10, 10, str, str+1, S("abcdefghij1"));
    test(S("abcdefghijklmnopqrst"), 10, 10, str, str+2, S("abcdefghij12"));
    test(S("abcdefghijklmnopqrst"), 10, 10, str, str+4, S("abcdefghij1234"));
    test(S("abcdefghijklmnopqrst"), 10, 10, str, str+5, S("abcdefghij12345"));
    test(S("abcdefghijklmnopqrst"), 10, 10, str, str+0, S("abcdefghij"));
    test(S("abcdefghijklmnopqrst"), 10, 10, str, str+1, S("abcdefghij1"));
    test(S("abcdefghijklmnopqrst"), 10, 10, str, str+5, S("abcdefghij12345"));
    test(S("abcdefghijklmnopqrst"), 10, 10, str, str+9, S("abcdefghij123456789"));
    test(S("abcdefghijklmnopqrst"), 10, 10, str, str+10, S("abcdefghij1234567890"));
    test(S("abcdefghijklmnopqrst"), 10, 10, str, str+0, S("abcdefghij"));
    test(S("abcdefghijklmnopqrst"), 10, 10, str, str+1, S("abcdefghij1"));
    test(S("abcdefghijklmnopqrst"), 10, 10, str, str+10, S("abcdefghij1234567890"));
    test(S("abcdefghijklmnopqrst"), 10, 10, str, str+19, S("abcdefghij1234567890123456789"));
    test(S("abcdefghijklmnopqrst"), 10, 10, str, str+20, S("abcdefghij12345678901234567890"));
    test(S("abcdefghijklmnopqrst"), 19, 0, str, str+0, S("abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 19, 0, str, str+0, S("abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 19, 0, str, str+1, S("abcdefghijklmnopqrs1t"));
    test(S("abcdefghijklmnopqrst"), 19, 0, str, str+2, S("abcdefghijklmnopqrs12t"));
    test(S("abcdefghijklmnopqrst"), 19, 0, str, str+4, S("abcdefghijklmnopqrs1234t"));
    test(S("abcdefghijklmnopqrst"), 19, 0, str, str+5, S("abcdefghijklmnopqrs12345t"));
    test(S("abcdefghijklmnopqrst"), 19, 0, str, str+0, S("abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 19, 0, str, str+1, S("abcdefghijklmnopqrs1t"));
    test(S("abcdefghijklmnopqrst"), 19, 0, str, str+5, S("abcdefghijklmnopqrs12345t"));
    test(S("abcdefghijklmnopqrst"), 19, 0, str, str+9, S("abcdefghijklmnopqrs123456789t"));
    test(S("abcdefghijklmnopqrst"), 19, 0, str, str+10, S("abcdefghijklmnopqrs1234567890t"));
    test(S("abcdefghijklmnopqrst"), 19, 0, str, str+0, S("abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 19, 0, str, str+1, S("abcdefghijklmnopqrs1t"));
    test(S("abcdefghijklmnopqrst"), 19, 0, str, str+10, S("abcdefghijklmnopqrs1234567890t"));
    test(S("abcdefghijklmnopqrst"), 19, 0, str, str+19, S("abcdefghijklmnopqrs1234567890123456789t"));
    test(S("abcdefghijklmnopqrst"), 19, 0, str, str+20, S("abcdefghijklmnopqrs12345678901234567890t"));
    test(S("abcdefghijklmnopqrst"), 19, 1, str, str+0, S("abcdefghijklmnopqrs"));
    test(S("abcdefghijklmnopqrst"), 19, 1, str, str+0, S("abcdefghijklmnopqrs"));
    test(S("abcdefghijklmnopqrst"), 19, 1, str, str+1, S("abcdefghijklmnopqrs1"));
    test(S("abcdefghijklmnopqrst"), 19, 1, str, str+2, S("abcdefghijklmnopqrs12"));
    test(S("abcdefghijklmnopqrst"), 19, 1, str, str+4, S("abcdefghijklmnopqrs1234"));
    test(S("abcdefghijklmnopqrst"), 19, 1, str, str+5, S("abcdefghijklmnopqrs12345"));
    test(S("abcdefghijklmnopqrst"), 19, 1, str, str+0, S("abcdefghijklmnopqrs"));
    test(S("abcdefghijklmnopqrst"), 19, 1, str, str+1, S("abcdefghijklmnopqrs1"));
    test(S("abcdefghijklmnopqrst"), 19, 1, str, str+5, S("abcdefghijklmnopqrs12345"));
    test(S("abcdefghijklmnopqrst"), 19, 1, str, str+9, S("abcdefghijklmnopqrs123456789"));
    test(S("abcdefghijklmnopqrst"), 19, 1, str, str+10, S("abcdefghijklmnopqrs1234567890"));
    test(S("abcdefghijklmnopqrst"), 19, 1, str, str+0, S("abcdefghijklmnopqrs"));
    test(S("abcdefghijklmnopqrst"), 19, 1, str, str+1, S("abcdefghijklmnopqrs1"));
    test(S("abcdefghijklmnopqrst"), 19, 1, str, str+10, S("abcdefghijklmnopqrs1234567890"));
    test(S("abcdefghijklmnopqrst"), 19, 1, str, str+19, S("abcdefghijklmnopqrs1234567890123456789"));
    test(S("abcdefghijklmnopqrst"), 19, 1, str, str+20, S("abcdefghijklmnopqrs12345678901234567890"));
    test(S("abcdefghijklmnopqrst"), 20, 0, str, str+0, S("abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 20, 0, str, str+0, S("abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 20, 0, str, str+1, S("abcdefghijklmnopqrst1"));
    test(S("abcdefghijklmnopqrst"), 20, 0, str, str+2, S("abcdefghijklmnopqrst12"));
    test(S("abcdefghijklmnopqrst"), 20, 0, str, str+4, S("abcdefghijklmnopqrst1234"));
    test(S("abcdefghijklmnopqrst"), 20, 0, str, str+5, S("abcdefghijklmnopqrst12345"));
    test(S("abcdefghijklmnopqrst"), 20, 0, str, str+0, S("abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 20, 0, str, str+1, S("abcdefghijklmnopqrst1"));
    test(S("abcdefghijklmnopqrst"), 20, 0, str, str+5, S("abcdefghijklmnopqrst12345"));
    test(S("abcdefghijklmnopqrst"), 20, 0, str, str+9, S("abcdefghijklmnopqrst123456789"));
    test(S("abcdefghijklmnopqrst"), 20, 0, str, str+10, S("abcdefghijklmnopqrst1234567890"));
    test(S("abcdefghijklmnopqrst"), 20, 0, str, str+0, S("abcdefghijklmnopqrst"));
    test(S("abcdefghijklmnopqrst"), 20, 0, str, str+1, S("abcdefghijklmnopqrst1"));
    test(S("abcdefghijklmnopqrst"), 20, 0, str, str+10, S("abcdefghijklmnopqrst1234567890"));
    test(S("abcdefghijklmnopqrst"), 20, 0, str, str+19, S("abcdefghijklmnopqrst1234567890123456789"));
    test(S("abcdefghijklmnopqrst"), 20, 0, str, str+20, S("abcdefghijklmnopqrst12345678901234567890"));

    return true;
}

template <class S>
bool test9() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  { // test iterator operations that throw
    typedef ThrowingIterator<char> TIter;
    typedef cpp17_input_iterator<TIter> IIter;
    const char* s = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    test_exceptions(S("abcdefghijklmnopqrst"), 10, 5, IIter(TIter(s, s+10, 4, TIter::TAIncrement)), IIter(TIter()));
    test_exceptions(S("abcdefghijklmnopqrst"), 10, 5, IIter(TIter(s, s+10, 5, TIter::TADereference)), IIter(TIter()));
    test_exceptions(S("abcdefghijklmnopqrst"), 10, 5, IIter(TIter(s, s+10, 6, TIter::TAComparison)), IIter(TIter()));

    test_exceptions(S("abcdefghijklmnopqrst"), 10, 5, TIter(s, s+10, 4, TIter::TAIncrement), TIter());
    test_exceptions(S("abcdefghijklmnopqrst"), 10, 5, TIter(s, s+10, 5, TIter::TADereference), TIter());
    test_exceptions(S("abcdefghijklmnopqrst"), 10, 5, TIter(s, s+10, 6, TIter::TAComparison), TIter());

    Widget w[100];
    test_exceptions(S("abcdefghijklmnopqrst"), 10, 5, w, w+100);
  }
#endif

  { // test replacing into self
    S s_short = "123/";
    S s_long  = "Lorem ipsum dolor sit amet, consectetur/";

    s_short.replace(s_short.begin(), s_short.begin(), s_short.begin(), s_short.end());
    assert(s_short == "123/123/");
    s_short.replace(s_short.begin(), s_short.begin(), s_short.begin(), s_short.end());
    assert(s_short == "123/123/123/123/");
    s_short.replace(s_short.begin(), s_short.begin(), s_short.begin(), s_short.end());
    assert(s_short == "123/123/123/123/123/123/123/123/");

    s_long.replace(s_long.begin(), s_long.begin(), s_long.begin(), s_long.end());
    assert(s_long == "Lorem ipsum dolor sit amet, consectetur/Lorem ipsum dolor sit amet, consectetur/");
  }

  { // test assigning a different type
    const uint8_t pc[] = "ABCD";
    uint8_t        p[] = "EFGH";

    S s;
    s.replace(s.begin(), s.end(), pc, pc + 4);
    assert(s == "ABCD");

    s.clear();
    s.replace(s.begin(), s.end(), p, p + 4);
    assert(s == "EFGH");
  }

  return true;
}

template <class S>
void test() {
  test0<S>();
  test1<S>();
  test2<S>();
  test3<S>();
  test4<S>();
  test5<S>();
  test6<S>();
  test7<S>();
  test8<S>();
  test9<S>();

#if TEST_STD_VER > 17
  // static_assert(test0<S>());
  // static_assert(test1<S>());
  // static_assert(test2<S>());
  // static_assert(test3<S>());
  // static_assert(test4<S>());
  // static_assert(test5<S>());
  // static_assert(test6<S>());
  // static_assert(test7<S>());
  // static_assert(test8<S>());
  // static_assert(test9<S>());
#endif
}

int main(int, char**)
{
  test<std::string>();
#if TEST_STD_VER >= 11
  test<std::basic_string<char, std::char_traits<char>, min_allocator<char>>>();
#endif

  return 0;
}
