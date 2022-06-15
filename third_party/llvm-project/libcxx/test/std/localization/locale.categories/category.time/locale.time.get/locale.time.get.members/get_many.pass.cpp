//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// class time_get<charT, InputIterator>

// iter_type
// get(iter_type s, iter_type end, ios_base& f, ios_base::iostate& err, tm *t,
//     const char_type *fmt, const char_type *fmtend) const;

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
        const char in[] = "2009 May 9, 10:27pm";
        const char fmt[] = "%Y %b %d, %I:%M%p";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get(I(in), I(in+sizeof(in)-1), ios, err, &t, fmt, fmt+sizeof(fmt)-1);
        assert(base(i) == in+sizeof(in)-1);
        assert(t.tm_year == 109);
        assert(t.tm_mon == 4);
        assert(t.tm_mday == 9);
        assert(t.tm_hour == 22);
        assert(t.tm_min == 27);
        assert(err == std::ios_base::eofbit);
    }
    {
        const char in[] = "10:27PM May 9, 2009";
        const char fmt[] = "%I:%M%p %b %d, %Y";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get(I(in), I(in+sizeof(in)-1), ios, err, &t, fmt, fmt+sizeof(fmt)-1);
        assert(base(i) == in+sizeof(in)-1);
        assert(t.tm_year == 109);
        assert(t.tm_mon == 4);
        assert(t.tm_mday == 9);
        assert(t.tm_hour == 22);
        assert(t.tm_min == 27);
        assert(err == std::ios_base::eofbit);
    }

  return 0;
}
