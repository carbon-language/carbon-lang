//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// NetBSD does not support LC_TIME at the moment
// XFAIL: netbsd

// XFAIL: LIBCXX-WINDOWS-FIXME

// REQUIRES: locale.en_US.UTF-8
// REQUIRES: locale.fr_FR.UTF-8

// <locale>

// template <class CharT, class OutputIterator = ostreambuf_iterator<CharT> >
// class time_put_byname
//     : public time_put<CharT, OutputIterator>
// {
// public:
//     explicit time_put_byname(const char* nm, size_t refs = 0);
//     explicit time_put_byname(const string& nm, size_t refs = 0);
//
// protected:
//     ~time_put_byname();
// };

// TODO: investigation needed
// XFAIL: target={{.*}}-linux-gnu{{.*}}

#include <locale>
#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"

#include "platform_support.h" // locale name macros

typedef std::time_put_byname<char, output_iterator<char*> > F;

class my_facet
    : public F
{
public:
    explicit my_facet(const std::string& nm, std::size_t refs = 0)
        : F(nm, refs) {}
};

int main(int, char**)
{
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
        const my_facet f(LOCALE_en_US_UTF_8, 1);
        std::string pat("Today is %A which is abbreviated %a.");
        iter = f.put(output_iterator<char*>(str), ios, '*', &t,
                     pat.data(), pat.data() + pat.size());
        std::string ex(str, iter.base());
        assert(ex == "Today is Saturday which is abbreviated Sat.");
    }
    {
        const my_facet f(LOCALE_fr_FR_UTF_8, 1);
        std::string pat("Today is %A which is abbreviated %a.");
        iter = f.put(output_iterator<char*>(str), ios, '*', &t,
                     pat.data(), pat.data() + pat.size());
        std::string ex(str, iter.base());
        assert((ex == "Today is Samedi which is abbreviated Sam.")||
               (ex == "Today is samedi which is abbreviated sam." ));
    }

  return 0;
}
