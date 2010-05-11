//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// template<class charT, class traits, class Allocator> 
//   basic_string<charT,traits,Allocator>
//   operator+(const basic_string<charT,traits,Allocator>& lhs, charT rhs);

// template<class charT, class traits, class Allocator> 
//   basic_string<charT,traits,Allocator>&&
//   operator+(basic_string<charT,traits,Allocator>&& lhs, charT rhs);

#include <string>
#include <cassert>

template <class S>
void
test0(const S& lhs, typename S::value_type rhs, const S& x)
{
    assert(lhs + rhs == x);
}

#ifdef _LIBCPP_MOVE

template <class S>
void
test1(S&& lhs, typename S::value_type rhs, const S& x)
{
    assert(move(lhs) + rhs == x);
}

#endif

typedef std::string S;

int main()
{
    test0(S(""), '1', S("1"));
    test0(S("abcde"), '1', S("abcde1"));
    test0(S("abcdefghij"), '1', S("abcdefghij1"));
    test0(S("abcdefghijklmnopqrst"), '1', S("abcdefghijklmnopqrst1"));

#ifdef _LIBCPP_MOVE

    test1(S(""), '1', S("1"));
    test1(S("abcde"), '1', S("abcde1"));
    test1(S("abcdefghij"), '1', S("abcdefghij1"));
    test1(S("abcdefghijklmnopqrst"), '1', S("abcdefghijklmnopqrst1"));

#endif
}
