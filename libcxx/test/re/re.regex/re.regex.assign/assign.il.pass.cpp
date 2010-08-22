//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <regex>

// template <class charT, class traits = regex_traits<charT>> class basic_regex;

// basic_regex&
//    assign(initializer_list<charT> il,
//           flag_type f = regex_constants::ECMAScript);

#include <regex>
#include <cassert>

int main()
{
#ifdef _LIBCPP_MOVE
    std::regex r2;
    r2.assign({'(', 'a', '(', '[', 'b', 'c', ']', ')', ')'});
    assert(r2.flags() == std::regex::ECMAScript);
    assert(r2.mark_count() == 2);

    r2.assign({'(', 'a', '(', '[', 'b', 'c', ']', ')', ')'}, std::regex::extended);
    assert(r2.flags() == std::regex::extended);
    assert(r2.mark_count() == 2);
#endif  // _LIBCPP_MOVE
}
