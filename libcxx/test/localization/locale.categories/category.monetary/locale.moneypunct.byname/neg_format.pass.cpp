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

// pattern neg_format() const;

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
        std::money_base::pattern p = f.neg_format();
        assert(p.field[0] == std::money_base::symbol);
        assert(p.field[1] == std::money_base::sign);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::value);
    }
    {
        Fnt f("C", 1);
        std::money_base::pattern p = f.neg_format();
        assert(p.field[0] == std::money_base::symbol);
        assert(p.field[1] == std::money_base::sign);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::value);
    }
    {
        Fwf f("C", 1);
        std::money_base::pattern p = f.neg_format();
        assert(p.field[0] == std::money_base::symbol);
        assert(p.field[1] == std::money_base::sign);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::value);
    }
    {
        Fwt f("C", 1);
        std::money_base::pattern p = f.neg_format();
        assert(p.field[0] == std::money_base::symbol);
        assert(p.field[1] == std::money_base::sign);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::value);
    }

    {
        Fnf f("en_US.UTF-8", 1);
        std::money_base::pattern p = f.neg_format();
        assert(p.field[0] == std::money_base::sign);
        assert(p.field[1] == std::money_base::symbol);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::value);
    }
    {
        Fnt f("en_US.UTF-8", 1);
        std::money_base::pattern p = f.neg_format();
        assert(p.field[0] == std::money_base::sign);
        assert(p.field[1] == std::money_base::symbol);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::value);
    }
    {
        Fwf f("en_US.UTF-8", 1);
        std::money_base::pattern p = f.neg_format();
        assert(p.field[0] == std::money_base::sign);
        assert(p.field[1] == std::money_base::symbol);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::value);
    }
    {
        Fwt f("en_US.UTF-8", 1);
        std::money_base::pattern p = f.neg_format();
        assert(p.field[0] == std::money_base::sign);
        assert(p.field[1] == std::money_base::symbol);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::value);
    }

    {
        Fnf f("fr_FR.UTF-8", 1);
        std::money_base::pattern p = f.neg_format();
        assert(p.field[0] == std::money_base::value);
        assert(p.field[1] == std::money_base::space);
        assert(p.field[2] == std::money_base::symbol);
        assert(p.field[3] == std::money_base::sign);
    }
    {
        Fnt f("fr_FR.UTF-8", 1);
        std::money_base::pattern p = f.neg_format();
        assert(p.field[0] == std::money_base::value);
        assert(p.field[1] == std::money_base::space);
        assert(p.field[2] == std::money_base::symbol);
        assert(p.field[3] == std::money_base::sign);
    }
    {
        Fwf f("fr_FR.UTF-8", 1);
        std::money_base::pattern p = f.neg_format();
        assert(p.field[0] == std::money_base::value);
        assert(p.field[1] == std::money_base::space);
        assert(p.field[2] == std::money_base::symbol);
        assert(p.field[3] == std::money_base::sign);
    }
    {
        Fwt f("fr_FR.UTF-8", 1);
        std::money_base::pattern p = f.neg_format();
        assert(p.field[0] == std::money_base::value);
        assert(p.field[1] == std::money_base::space);
        assert(p.field[2] == std::money_base::symbol);
        assert(p.field[3] == std::money_base::sign);
    }

    {
        Fnf f("ru_RU.UTF-8", 1);
        std::money_base::pattern p = f.neg_format();
        assert(p.field[0] == std::money_base::sign);
        assert(p.field[1] == std::money_base::value);
        assert(p.field[2] == std::money_base::space);
        assert(p.field[3] == std::money_base::symbol);
    }
    {
        Fnt f("ru_RU.UTF-8", 1);
        std::money_base::pattern p = f.neg_format();
        assert(p.field[0] == std::money_base::sign);
        assert(p.field[1] == std::money_base::value);
        assert(p.field[2] == std::money_base::space);
        assert(p.field[3] == std::money_base::symbol);
    }
    {
        Fwf f("ru_RU.UTF-8", 1);
        std::money_base::pattern p = f.neg_format();
        assert(p.field[0] == std::money_base::sign);
        assert(p.field[1] == std::money_base::value);
        assert(p.field[2] == std::money_base::space);
        assert(p.field[3] == std::money_base::symbol);
    }
    {
        Fwt f("ru_RU.UTF-8", 1);
        std::money_base::pattern p = f.neg_format();
        assert(p.field[0] == std::money_base::sign);
        assert(p.field[1] == std::money_base::value);
        assert(p.field[2] == std::money_base::space);
        assert(p.field[3] == std::money_base::symbol);
    }

    {
        Fnf f("zh_CN.UTF-8", 1);
        std::money_base::pattern p = f.neg_format();
        assert(p.field[0] == std::money_base::symbol);
        assert(p.field[1] == std::money_base::sign);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::value);
    }
    {
        Fnt f("zh_CN.UTF-8", 1);
        std::money_base::pattern p = f.neg_format();
        assert(p.field[0] == std::money_base::symbol);
        assert(p.field[1] == std::money_base::sign);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::value);
    }
    {
        Fwf f("zh_CN.UTF-8", 1);
        std::money_base::pattern p = f.neg_format();
        assert(p.field[0] == std::money_base::symbol);
        assert(p.field[1] == std::money_base::sign);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::value);
    }
    {
        Fwt f("zh_CN.UTF-8", 1);
        std::money_base::pattern p = f.neg_format();
        assert(p.field[0] == std::money_base::symbol);
        assert(p.field[1] == std::money_base::sign);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::value);
    }
}
