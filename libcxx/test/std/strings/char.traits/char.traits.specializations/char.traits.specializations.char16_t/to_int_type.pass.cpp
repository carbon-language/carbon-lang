//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// template<> struct char_traits<char16_t>

// static constexpr int_type to_int_type(char_type c);

#include <string>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
#if __cplusplus >= 201103L
    assert(std::char_traits<char16_t>::to_int_type(u'a') == u'a');
    assert(std::char_traits<char16_t>::to_int_type(u'A') == u'A');
#endif
    assert(std::char_traits<char16_t>::to_int_type(0) == 0);
#endif  // _LIBCPP_HAS_NO_UNICODE_CHARS
}
