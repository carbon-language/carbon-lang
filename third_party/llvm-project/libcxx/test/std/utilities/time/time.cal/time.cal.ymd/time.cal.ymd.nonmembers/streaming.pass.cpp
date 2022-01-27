//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: *

// <chrono>
// class year_month_day;

// template<class charT, class traits>
//     basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const year_month_day& ym);
//
// Returns: os << ym.year() << '/' << ym.month().
//
//
// template<class charT, class traits>
//     basic_ostream<charT, traits>&
//     to_stream(basic_ostream<charT, traits>& os, const charT* fmt, const year_month_day& ym);
//
// Effects: Streams ym into os using the format specified by the NTCTS fmt. fmt encoding follows the rules specified in 25.11.
//
// template<class charT, class traits, class Alloc = allocator<charT>>
//     basic_istream<charT, traits>&
//   from_stream(basic_istream<charT, traits>& is, const charT* fmt,
//               year_month_day& ym, basic_string<charT, traits, Alloc>* abbrev = nullptr,
//               minutes* offset = nullptr);
//
// Effects: Attempts to parse the input stream is into the year_month_day ym using the format
//         flags given in the NTCTS fmt as specified in 25.12. If the parse fails to decode
//         a valid year_month_day, is.setstate(ios_- base::failbit) shall be called and ym shall
//         not be modified. If %Z is used and successfully parsed, that value will be assigned
//         to *abbrev if abbrev is non-null. If %z (or a modified variant) is used and
//         successfully parsed, that value will be assigned to *offset if offset is non-null.



#include <chrono>
#include <type_traits>
#include <cassert>
#include <iostream>

#include "test_macros.h"

int main(int, char**)
{
    using year_month_day = std::chrono::year_month_day;
    using year           = std::chrono::year;
    using month          = std::chrono::month;
    using day            = std::chrono::day;

    std::cout << year_month_day{year{2018}, month{3}, day{12}};

  return 0;
}
