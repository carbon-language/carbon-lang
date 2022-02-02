//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// class time_get<charT, InputIterator>

// iter_type get_year(iter_type s, iter_type end, ios_base& str,
//                    ios_base::iostate& err, tm* t) const;

#include <locale>
#include <cassert>
#include <ios>
#include "test_macros.h"
#include "test_iterators.h"

typedef cpp17_input_iterator<const char*> I;

typedef std::time_get<char, I> F;

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
    std::ios ios(0);
    std::ios_base::iostate err;
    std::tm t;
    {
        const char in[] = "0";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_year(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+sizeof(in)-1);
        assert(t.tm_year == 100);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "00";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_year(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+sizeof(in)-1);
        assert(t.tm_year == 100);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "1";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_year(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+sizeof(in)-1);
        assert(t.tm_year == 101);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "68";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_year(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+sizeof(in)-1);
        assert(t.tm_year == 168);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "69";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_year(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+sizeof(in)-1);
        assert(t.tm_year == 69);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "99";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_year(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+sizeof(in)-1);
        assert(t.tm_year == 99);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "100";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_year(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+sizeof(in)-1);
        assert(t.tm_year == -1800);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "1900";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_year(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+sizeof(in)-1);
        assert(t.tm_year == 0);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "1968";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_year(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+sizeof(in)-1);
        assert(t.tm_year == 68);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "2000";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_year(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+sizeof(in)-1);
        assert(t.tm_year == 100);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "2999c";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_year(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+sizeof(in)-2);
        assert(t.tm_year == 1099);
        assert(err == std::ios_base::goodbit);
    }

  return 0;
}
