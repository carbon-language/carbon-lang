//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// void pop_back();

#if _LIBCPP_DEBUG >= 1
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))
#endif

#include <string>
#include <cassert>

#include "min_allocator.h"

template <class S>
void
test(S s, S expected)
{
    s.pop_back();
    assert(s.__invariants());
    assert(s == expected);
}

int main()
{
    {
    typedef std::string S;
    test(S("abcde"), S("abcd"));
    test(S("abcdefghij"), S("abcdefghi"));
    test(S("abcdefghijklmnopqrst"), S("abcdefghijklmnopqrs"));
    }
#if __cplusplus >= 201103L
    {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    test(S("abcde"), S("abcd"));
    test(S("abcdefghij"), S("abcdefghi"));
    test(S("abcdefghijklmnopqrst"), S("abcdefghijklmnopqrs"));
    }
#endif
#if _LIBCPP_DEBUG >= 1
    {
        std::string s;
        s.pop_back();
        assert(false);
    }
#endif        
}
