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

// string_type curr_symbol() const;

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
        assert(f.curr_symbol() == std::string());
    }
    {
        Fnt f("C", 1);
        assert(f.curr_symbol() == std::string());
    }
    {
        Fwf f("C", 1);
        assert(f.curr_symbol() == std::wstring());
    }
    {
        Fwt f("C", 1);
        assert(f.curr_symbol() == std::wstring());
    }

    {
        Fnf f("en_US.UTF-8", 1);
        assert(f.curr_symbol() == "$");
    }
    {
        Fnt f("en_US.UTF-8", 1);
        assert(f.curr_symbol() == "USD ");
    }
    {
        Fwf f("en_US.UTF-8", 1);
        assert(f.curr_symbol() == L"$");
    }
    {
        Fwt f("en_US.UTF-8", 1);
        assert(f.curr_symbol() == L"USD ");
    }

    {
        Fnf f("fr_FR.UTF-8", 1);
        assert(f.curr_symbol() == "Eu");
    }
    {
        Fnt f("fr_FR.UTF-8", 1);
        assert(f.curr_symbol() == "EUR ");
    }
    {
        Fwf f("fr_FR.UTF-8", 1);
        assert(f.curr_symbol() == L"Eu");
    }
    {
        Fwt f("fr_FR.UTF-8", 1);
        assert(f.curr_symbol() == L"EUR ");
    }

    {
        Fnf f("ru_RU.UTF-8", 1);
        assert(f.curr_symbol() == "\xD1\x80\xD1\x83\xD0\xB1"".");
    }
    {
        Fnt f("ru_RU.UTF-8", 1);
        assert(f.curr_symbol() == "RUB ");
    }
    {
        Fwf f("ru_RU.UTF-8", 1);
        assert(f.curr_symbol() == L"\x440\x443\x431"".");
    }
    {
        Fwt f("ru_RU.UTF-8", 1);
        assert(f.curr_symbol() == L"RUB ");
    }

    {
        Fnf f("zh_CN.UTF-8", 1);
        assert(f.curr_symbol() == "\xEF\xBF\xA5");
    }
    {
        Fnt f("zh_CN.UTF-8", 1);
        assert(f.curr_symbol() == "CNY ");
    }
    {
        Fwf f("zh_CN.UTF-8", 1);
        assert(f.curr_symbol() == L"\xFFE5");
    }
    {
        Fwt f("zh_CN.UTF-8", 1);
        assert(f.curr_symbol() == L"CNY ");
    }
}
