//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// XFAIL: darwin
//
// NetBSD does not support LC_MONETARY at the moment
// XFAIL: netbsd

// XFAIL: LIBCXX-WINDOWS-FIXME

// REQUIRES: locale.en_US.UTF-8
// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ru_RU.UTF-8
// REQUIRES: locale.zh_CN.UTF-8

// <locale>

// class moneypunct_byname<charT, International>

// pattern pos_format() const;

#include <locale>
#include <limits>
#include <cassert>

#include "test_macros.h"
#include "platform_support.h" // locale name macros

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

int main(int, char**)
{
    {
        Fnf f("C", 1);
        std::money_base::pattern p = f.pos_format();
        assert(p.field[0] == std::money_base::symbol);
        assert(p.field[1] == std::money_base::sign);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::value);
    }
    {
        Fnt f("C", 1);
        std::money_base::pattern p = f.pos_format();
        assert(p.field[0] == std::money_base::symbol);
        assert(p.field[1] == std::money_base::sign);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::value);
    }
    {
        Fwf f("C", 1);
        std::money_base::pattern p = f.pos_format();
        assert(p.field[0] == std::money_base::symbol);
        assert(p.field[1] == std::money_base::sign);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::value);
    }
    {
        Fwt f("C", 1);
        std::money_base::pattern p = f.pos_format();
        assert(p.field[0] == std::money_base::symbol);
        assert(p.field[1] == std::money_base::sign);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::value);
    }

    {
        Fnf f(LOCALE_en_US_UTF_8, 1);
        std::money_base::pattern p = f.pos_format();
        assert(p.field[0] == std::money_base::sign);
        assert(p.field[1] == std::money_base::symbol);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::value);
    }
    {
        Fnt f(LOCALE_en_US_UTF_8, 1);
        std::money_base::pattern p = f.pos_format();
        assert(p.field[0] == std::money_base::sign);
        assert(p.field[1] == std::money_base::symbol);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::value);
    }
    {
        Fwf f(LOCALE_en_US_UTF_8, 1);
        std::money_base::pattern p = f.pos_format();
        assert(p.field[0] == std::money_base::sign);
        assert(p.field[1] == std::money_base::symbol);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::value);
    }
    {
        Fwt f(LOCALE_en_US_UTF_8, 1);
        std::money_base::pattern p = f.pos_format();
        assert(p.field[0] == std::money_base::sign);
        assert(p.field[1] == std::money_base::symbol);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::value);
    }

    {
        Fnf f(LOCALE_fr_FR_UTF_8, 1);
        std::money_base::pattern p = f.pos_format();
        assert(p.field[0] == std::money_base::sign);
        assert(p.field[1] == std::money_base::value);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::symbol);
    }
    {
        Fnt f(LOCALE_fr_FR_UTF_8, 1);
        std::money_base::pattern p = f.pos_format();
        assert(p.field[0] == std::money_base::sign);
        assert(p.field[1] == std::money_base::value);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::symbol);
    }
    {
        Fwf f(LOCALE_fr_FR_UTF_8, 1);
        std::money_base::pattern p = f.pos_format();
        assert(p.field[0] == std::money_base::sign);
        assert(p.field[1] == std::money_base::value);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::symbol);
    }
    {
        Fwt f(LOCALE_fr_FR_UTF_8, 1);
        std::money_base::pattern p = f.pos_format();
        assert(p.field[0] == std::money_base::sign);
        assert(p.field[1] == std::money_base::value);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::symbol);
    }

    {
        Fnf f(LOCALE_ru_RU_UTF_8, 1);
        std::money_base::pattern p = f.pos_format();
        assert(p.field[0] == std::money_base::sign);
        assert(p.field[1] == std::money_base::value);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::symbol);
    }
    {
        Fnt f(LOCALE_ru_RU_UTF_8, 1);
        std::money_base::pattern p = f.pos_format();
        assert(p.field[0] == std::money_base::sign);
        assert(p.field[1] == std::money_base::value);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::symbol);
    }
    {
        Fwf f(LOCALE_ru_RU_UTF_8, 1);
        std::money_base::pattern p = f.pos_format();
        assert(p.field[0] == std::money_base::sign);
        assert(p.field[1] == std::money_base::value);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::symbol);
    }
    {
        Fwt f(LOCALE_ru_RU_UTF_8, 1);
        std::money_base::pattern p = f.pos_format();
        assert(p.field[0] == std::money_base::sign);
        assert(p.field[1] == std::money_base::value);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::symbol);
    }

    {
        Fnf f(LOCALE_zh_CN_UTF_8, 1);
        std::money_base::pattern p = f.pos_format();
        assert(p.field[0] == std::money_base::symbol);
        assert(p.field[1] == std::money_base::sign);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::value);
    }
    {
        Fnt f(LOCALE_zh_CN_UTF_8, 1);
        std::money_base::pattern p = f.pos_format();
        assert(p.field[0] == std::money_base::sign);
        assert(p.field[1] == std::money_base::symbol);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::value);
    }
    {
        Fwf f(LOCALE_zh_CN_UTF_8, 1);
        std::money_base::pattern p = f.pos_format();
        assert(p.field[0] == std::money_base::symbol);
        assert(p.field[1] == std::money_base::sign);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::value);
    }
    {
        Fwt f(LOCALE_zh_CN_UTF_8, 1);
        std::money_base::pattern p = f.pos_format();
        assert(p.field[0] == std::money_base::sign);
        assert(p.field[1] == std::money_base::symbol);
        assert(p.field[2] == std::money_base::none);
        assert(p.field[3] == std::money_base::value);
    }

  return 0;
}
