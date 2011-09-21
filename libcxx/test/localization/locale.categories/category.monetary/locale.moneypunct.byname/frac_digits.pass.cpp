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

// int frac_digits() const;

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
        assert(f.frac_digits() == 0);
    }
    {
        Fnt f("C", 1);
        assert(f.frac_digits() == 0);
    }
    {
        Fwf f("C", 1);
        assert(f.frac_digits() == 0);
    }
    {
        Fwt f("C", 1);
        assert(f.frac_digits() == 0);
    }

    {
        Fnf f("en_US.UTF-8", 1);
        assert(f.frac_digits() == 2);
    }
    {
        Fnt f("en_US.UTF-8", 1);
        assert(f.frac_digits() == 2);
    }
    {
        Fwf f("en_US.UTF-8", 1);
        assert(f.frac_digits() == 2);
    }
    {
        Fwt f("en_US.UTF-8", 1);
        assert(f.frac_digits() == 2);
    }

    {
        Fnf f("fr_FR.UTF-8", 1);
        assert(f.frac_digits() == 2);
    }
    {
        Fnt f("fr_FR.UTF-8", 1);
        assert(f.frac_digits() == 2);
    }
    {
        Fwf f("fr_FR.UTF-8", 1);
        assert(f.frac_digits() == 2);
    }
    {
        Fwt f("fr_FR.UTF-8", 1);
        assert(f.frac_digits() == 2);
    }

    {
        Fnf f("ru_RU.UTF-8", 1);
        assert(f.frac_digits() == 2);
    }
    {
        Fnt f("ru_RU.UTF-8", 1);
        assert(f.frac_digits() == 2);
    }
    {
        Fwf f("ru_RU.UTF-8", 1);
        assert(f.frac_digits() == 2);
    }
    {
        Fwt f("ru_RU.UTF-8", 1);
        assert(f.frac_digits() == 2);
    }

    {
        Fnf f("zh_CN.UTF-8", 1);
        assert(f.frac_digits() == 2);
    }
    {
        Fnt f("zh_CN.UTF-8", 1);
        assert(f.frac_digits() == 2);
    }
    {
        Fwf f("zh_CN.UTF-8", 1);
        assert(f.frac_digits() == 2);
    }
    {
        Fwt f("zh_CN.UTF-8", 1);
        assert(f.frac_digits() == 2);
    }
}
