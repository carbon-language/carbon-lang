//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// class time_put<charT, OutputIterator>

// iter_type put(iter_type s, ios_base& str, char_type fill, const tm* t,
//               const charT* pattern, const charT* pat_end) const;

#include <locale>
#include <cassert>
#include <ios>
#include "test_macros.h"
#include "test_iterators.h"

typedef std::time_put<char, output_iterator<char*> > F;

class my_facet
    : public F
{
public:
    explicit my_facet(std::size_t refs = 0)
        : F(refs) {}
};

int main(int, char**)
{
    const my_facet f(1);
    char str[200];
    output_iterator<char*> iter;
    tm t;
    t.tm_sec = 6;
    t.tm_min = 3;
    t.tm_hour = 13;
    t.tm_mday = 2;
    t.tm_mon = 4;
    t.tm_year = 109;
    t.tm_wday = 6;
    t.tm_yday = -1;
    t.tm_isdst = 1;
    std::ios ios(0);
    {
        std::string pat("Today is %A which is abbreviated %a.");
        iter = f.put(output_iterator<char*>(str), ios, '*', &t,
                     pat.data(), pat.data() + pat.size());
        std::string ex(str, iter.base());
        assert(ex == "Today is Saturday which is abbreviated Sat.");
    }
    {
        std::string pat("The number of the month is %Om.");
        iter = f.put(output_iterator<char*>(str), ios, '*', &t,
                     pat.data(), pat.data() + pat.size());
        std::string ex(str, iter.base());
        assert(ex == "The number of the month is 05.");
    }

  return 0;
}
