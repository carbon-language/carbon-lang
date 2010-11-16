//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <locale>

// class moneypunct_byname<charT, International>

// charT thousands_sep() const;

#include <locale>
#include <limits>
#include <cassert>

class Fnf
    : public std::moneypunct_byname<char, false>
{
public:
    explicit Fnf(const std::string& nm, std::size_t refs = 0)
        : std::moneypunct_byname<char, false>(nm, refs) {}
};

class Fnt
    : public std::moneypunct_byname<char, true>
{
public:
    explicit Fnt(const std::string& nm, std::size_t refs = 0)
        : std::moneypunct_byname<char, true>(nm, refs) {}
};

class Fwf
    : public std::moneypunct_byname<wchar_t, false>
{
public:
    explicit Fwf(const std::string& nm, std::size_t refs = 0)
        : std::moneypunct_byname<wchar_t, false>(nm, refs) {}
};

class Fwt
    : public std::moneypunct_byname<wchar_t, true>
{
public:
    explicit Fwt(const std::string& nm, std::size_t refs = 0)
        : std::moneypunct_byname<wchar_t, true>(nm, refs) {}
};

int main()
{
    {
        Fnf f("C", 1);
        assert(f.thousands_sep() == std::numeric_limits<char>::max());
    }
    {
        Fnt f("C", 1);
        assert(f.thousands_sep() == std::numeric_limits<char>::max());
    }
    {
        Fwf f("C", 1);
        assert(f.thousands_sep() == std::numeric_limits<wchar_t>::max());
    }
    {
        Fwt f("C", 1);
        assert(f.thousands_sep() == std::numeric_limits<wchar_t>::max());
    }

    {
        Fnf f("en_US", 1);
        assert(f.thousands_sep() == ',');
    }
    {
        Fnt f("en_US", 1);
        assert(f.thousands_sep() == ',');
    }
    {
        Fwf f("en_US", 1);
        assert(f.thousands_sep() == L',');
    }
    {
        Fwt f("en_US", 1);
        assert(f.thousands_sep() == L',');
    }

    {
        Fnf f("fr_FR", 1);
        assert(f.thousands_sep() == ' ');
    }
    {
        Fnt f("fr_FR", 1);
        assert(f.thousands_sep() == ' ');
    }
    {
        Fwf f("fr_FR", 1);
        assert(f.thousands_sep() == L' ');
    }
    {
        Fwt f("fr_FR", 1);
        assert(f.thousands_sep() == L' ');
    }

    {
        Fnf f("ru_RU", 1);
        assert(f.thousands_sep() == ' ');
    }
    {
        Fnt f("ru_RU", 1);
        assert(f.thousands_sep() == ' ');
    }
    {
        Fwf f("ru_RU", 1);
        assert(f.thousands_sep() == L' ');
    }
    {
        Fwt f("ru_RU", 1);
        assert(f.thousands_sep() == L' ');
    }

    {
        Fnf f("zh_CN", 1);
        assert(f.thousands_sep() == ',');
    }
    {
        Fnt f("zh_CN", 1);
        assert(f.thousands_sep() == ',');
    }
    {
        Fwf f("zh_CN", 1);
        assert(f.thousands_sep() == L',');
    }
    {
        Fwt f("zh_CN", 1);
        assert(f.thousands_sep() == L',');
    }
}
