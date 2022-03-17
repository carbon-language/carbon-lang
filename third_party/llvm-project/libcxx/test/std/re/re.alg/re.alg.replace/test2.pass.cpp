//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// template <class OutputIterator, class BidirectionalIterator,
//           class traits, class charT, class ST, class SA>
//     OutputIterator
//     regex_replace(OutputIterator out,
//                   BidirectionalIterator first, BidirectionalIterator last,
//                   const basic_regex<charT, traits>& e,
//                   const charT* fmt,
//                   regex_constants::match_flag_type flags =
//                                              regex_constants::match_default);

#include <regex>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

int main(int, char**)
{
    {
        std::regex phone_numbers("\\d{3}-\\d{4}");
        const char phone_book[] = "555-1234, 555-2345, 555-3456";
        typedef cpp17_output_iterator<char*> Out;
        typedef bidirectional_iterator<const char*> Bi;
        char buf[100] = {0};
        Out r = std::regex_replace(Out(buf), Bi(std::begin(phone_book)),
                                    Bi(std::end(phone_book)-1), phone_numbers,
                                    "123-$&");
        assert(r.base() == buf+40);
        assert(buf == std::string("123-555-1234, 123-555-2345, 123-555-3456"));
    }
    {
        std::regex phone_numbers("\\d{3}-\\d{4}");
        const char phone_book[] = "555-1234, 555-2345, 555-3456";
        typedef cpp17_output_iterator<char*> Out;
        typedef bidirectional_iterator<const char*> Bi;
        char buf[100] = {0};
        Out r = std::regex_replace(Out(buf), Bi(std::begin(phone_book)),
                                    Bi(std::end(phone_book)-1), phone_numbers,
                                    "123-$&",
                                    std::regex_constants::format_sed);
        assert(r.base() == buf+43);
        assert(buf == std::string("123-$555-1234, 123-$555-2345, 123-$555-3456"));
    }
    {
        std::regex phone_numbers("\\d{3}-\\d{4}");
        const char phone_book[] = "555-1234, 555-2345, 555-3456";
        typedef cpp17_output_iterator<char*> Out;
        typedef bidirectional_iterator<const char*> Bi;
        char buf[100] = {0};
        Out r = std::regex_replace(Out(buf), Bi(std::begin(phone_book)),
                                    Bi(std::end(phone_book)-1), phone_numbers,
                                    "123-&",
                                    std::regex_constants::format_sed);
        assert(r.base() == buf+40);
        assert(buf == std::string("123-555-1234, 123-555-2345, 123-555-3456"));
    }
    {
        std::regex phone_numbers("\\d{3}-\\d{4}");
        const char phone_book[] = "555-1234, 555-2345, 555-3456";
        typedef cpp17_output_iterator<char*> Out;
        typedef bidirectional_iterator<const char*> Bi;
        char buf[100] = {0};
        Out r = std::regex_replace(Out(buf), Bi(std::begin(phone_book)),
                                    Bi(std::end(phone_book)-1), phone_numbers,
                                    "123-$&",
                                    std::regex_constants::format_no_copy);
        assert(r.base() == buf+36);
        assert(buf == std::string("123-555-1234123-555-2345123-555-3456"));
    }
    {
        std::regex phone_numbers("\\d{3}-\\d{4}");
        const char phone_book[] = "555-1234, 555-2345, 555-3456";
        typedef cpp17_output_iterator<char*> Out;
        typedef bidirectional_iterator<const char*> Bi;
        char buf[100] = {0};
        Out r = std::regex_replace(Out(buf), Bi(std::begin(phone_book)),
                                    Bi(std::end(phone_book)-1), phone_numbers,
                                    "123-$&",
                                    std::regex_constants::format_first_only);
        assert(r.base() == buf+32);
        assert(buf == std::string("123-555-1234, 555-2345, 555-3456"));
    }
    {
        std::regex phone_numbers("\\d{3}-\\d{4}");
        const char phone_book[] = "555-1234, 555-2345, 555-3456";
        typedef cpp17_output_iterator<char*> Out;
        typedef bidirectional_iterator<const char*> Bi;
        char buf[100] = {0};
        Out r = std::regex_replace(Out(buf), Bi(std::begin(phone_book)),
                                    Bi(std::end(phone_book)-1), phone_numbers,
                                    "123-$&",
                                    std::regex_constants::format_first_only |
                                    std::regex_constants::format_no_copy);
        assert(r.base() == buf+12);
        assert(buf == std::string("123-555-1234"));
    }

  return 0;
}
