//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <locale>

// class time_get_byname<charT, InputIterator>

// iter_type
// get_time(iter_type s, iter_type end, ios_base& str,
//          ios_base::iostate& err, tm* t) const;

#include <locale>
#include <cassert>
#include "iterators.h"

typedef input_iterator<const char*> I;

typedef std::time_get_byname<char, I> F;

class my_facet
    : public F
{
public:
    explicit my_facet(const std::string& nm, std::size_t refs = 0)
        : F(nm, refs) {}
};

int main()
{
    std::ios ios(0);
    std::ios_base::iostate err;
    std::tm t;
    {
        const my_facet f("en_US", 1);
        const char in[] = "13:14:15";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_time(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+sizeof(in)-1);
        assert(t.tm_hour == 13);
        assert(t.tm_min == 14);
        assert(t.tm_sec == 15);
        assert(err == std::ios_base::eofbit);
    }
    {
        const my_facet f("fr_FR", 1);
        const char in[] = "13:14:15";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_time(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+sizeof(in)-1);
        assert(t.tm_hour == 13);
        assert(t.tm_min == 14);
        assert(t.tm_sec == 15);
        assert(err == std::ios_base::eofbit);
    }
    {
        const my_facet f("ru_RU", 1);
        const char in[] = "13:14:15";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_time(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+sizeof(in)-1);
        assert(t.tm_hour == 13);
        assert(t.tm_min == 14);
        assert(t.tm_sec == 15);
        assert(err == std::ios_base::eofbit);
    }
    {
        const my_facet f("zh_CN", 1);
        const char in[] = "13:14:15";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_time(I(in), I(in+sizeof(in)-1), ios, err, &t);
        assert(i.base() == in+sizeof(in)-1);
        assert(t.tm_hour == 13);
        assert(t.tm_min == 14);
        assert(t.tm_sec == 15);
        assert(err == std::ios_base::eofbit);
    }
}
