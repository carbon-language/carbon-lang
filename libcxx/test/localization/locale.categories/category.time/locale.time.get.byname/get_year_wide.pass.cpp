//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <locale>

// class time_get_byname<charT, InputIterator>

// iter_type get_year(iter_type s, iter_type end, ios_base& str,
//                    ios_base::iostate& err, tm* t) const;

#include <locale>
#include <cassert>
#include "iterators.h"

typedef input_iterator<const wchar_t*> I;

typedef std::time_get_byname<wchar_t, I> F;

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
        const wchar_t in[] = L"2009";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_year(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(i.base() == in+sizeof(in)/sizeof(in[0])-1);
        assert(t.tm_year == 109);
        assert(err == std::ios_base::eofbit);
    }
    {
        const my_facet f("fr_FR", 1);
        const wchar_t in[] = L"2009";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_year(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(i.base() == in+sizeof(in)/sizeof(in[0])-1);
        assert(t.tm_year == 109);
        assert(err == std::ios_base::eofbit);
    }
    {
        const my_facet f("ru_RU", 1);
        const wchar_t in[] = L"2009";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_year(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(i.base() == in+sizeof(in)/sizeof(in[0])-1);
        assert(t.tm_year == 109);
        assert(err == std::ios_base::eofbit);
    }
    {
        const my_facet f("zh_CN", 1);
        const wchar_t in[] = L"2009";
        err = std::ios_base::goodbit;
        t = std::tm();
        I i = f.get_year(I(in), I(in+sizeof(in)/sizeof(in[0])-1), ios, err, &t);
        assert(i.base() == in+sizeof(in)/sizeof(in[0])-1);
        assert(t.tm_year == 109);
        assert(err == std::ios_base::eofbit);
    }
}
