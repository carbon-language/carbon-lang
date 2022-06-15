//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: locale.en_US.UTF-8
// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ru_RU.UTF-8
// REQUIRES: locale.zh_CN.UTF-8

// <locale>

// class time_get_byname<charT, InputIterator>

// iter_type
// get_time(iter_type s, iter_type end, ios_base& str,
//          ios_base::iostate& err, tm* t) const;

#include <locale>
#include <cassert>
#include <ios>
#include "test_macros.h"
#include "test_iterators.h"

#include "platform_support.h" // locale name macros

typedef cpp17_input_iterator<const char*> I;

typedef std::time_get_byname<char, I> F;

class my_facet
    : public F
{
public:
    explicit my_facet(const std::string& nm, std::size_t refs = 0)
        : F(nm, refs) {}
};

int main(int, char**)
{
    std::ios ios(0);
    std::ios_base::iostate err;
    std::tm t;
    {
        const my_facet f(LOCALE_en_US_UTF_8, 1);
        const char in[] = "13:14:15";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_time(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(base(i) == in+sizeof(in)-1);
        assert(t.tm_hour == 13);
        assert(t.tm_min == 14);
        assert(t.tm_sec == 15);
        assert(err == std::ios_base::eofbit);
    }
    {
        const my_facet f(LOCALE_fr_FR_UTF_8, 1);
        const char in[] = "13:14:15";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_time(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(base(i) == in+sizeof(in)-1);
        assert(t.tm_hour == 13);
        assert(t.tm_min == 14);
        assert(t.tm_sec == 15);
        assert(err == std::ios_base::eofbit);
    }
    {
        const my_facet f(LOCALE_ru_RU_UTF_8, 1);
        const char in[] = "13:14:15";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_time(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(base(i) == in+sizeof(in)-1);
        assert(t.tm_hour == 13);
        assert(t.tm_min == 14);
        assert(t.tm_sec == 15);
        assert(err == std::ios_base::eofbit);
    }
    {
        const my_facet f(LOCALE_zh_CN_UTF_8, 1);
        const char in[] = "13:14:15";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_time(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(base(i) == in+sizeof(in)-1);
        assert(t.tm_hour == 13);
        assert(t.tm_min == 14);
        assert(t.tm_sec == 15);
        assert(err == std::ios_base::eofbit);
    }

  return 0;
}
