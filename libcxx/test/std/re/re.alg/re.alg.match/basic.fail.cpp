//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <regex>

//   template <class ST, class SA, class Allocator, class charT, class traits> 
//   bool regex_match(const basic_string<charT, ST, SA>&&, 
//                    match_results<
//                      typename basic_string<charT, ST, SA>::const_iterator, 
//                      Allocator>&, 
//                    const basic_regex<charT, traits>&, 
//                    regex_constants::match_flag_type = 
//                      regex_constants::match_default) = delete;

#include <__config>

#if _LIBCPP_STD_VER <= 11
#error
#else

#include <regex>
#include <cassert>

int main()
{
    {
        std::smatch m;
        std::regex re{"*"};
        std::regex_match(std::string("abcde"), m, re);
    }
}
#endif
