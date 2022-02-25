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

// class moneypunct_byname<charT, International>

// charT decimal_point() const;

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
        assert(f.decimal_point() == std::numeric_limits<char>::max());
    }
    {
        Fnt f("C", 1);
        assert(f.decimal_point() == std::numeric_limits<char>::max());
    }
    {
        Fwf f("C", 1);
        assert(f.decimal_point() == std::numeric_limits<wchar_t>::max());
    }
    {
        Fwt f("C", 1);
        assert(f.decimal_point() == std::numeric_limits<wchar_t>::max());
    }

    {
        Fnf f(LOCALE_en_US_UTF_8, 1);
        assert(f.decimal_point() == '.');
    }
    {
        Fnt f(LOCALE_en_US_UTF_8, 1);
        assert(f.decimal_point() == '.');
    }
    {
        Fwf f(LOCALE_en_US_UTF_8, 1);
        assert(f.decimal_point() == L'.');
    }
    {
        Fwt f(LOCALE_en_US_UTF_8, 1);
        assert(f.decimal_point() == L'.');
    }

    {
        Fnf f(LOCALE_fr_FR_UTF_8, 1);
        assert(f.decimal_point() == ',');
    }
    {
        Fnt f(LOCALE_fr_FR_UTF_8, 1);
        assert(f.decimal_point() == ',');
    }
    {
        Fwf f(LOCALE_fr_FR_UTF_8, 1);
        assert(f.decimal_point() == L',');
    }
    {
        Fwt f(LOCALE_fr_FR_UTF_8, 1);
        assert(f.decimal_point() == L',');
    }
// GLIBC 2.23 uses '.' as the decimal point while other C libraries use ','
// GLIBC 2.27 corrects this
#if defined(_CS_GNU_LIBC_VERSION)
    const char sep = glibc_version_less_than("2.27") ? '.' : ',';
    const wchar_t wsep = glibc_version_less_than("2.27") ? L'.' : L',';
#else
    const char sep = ',';
    const wchar_t wsep = L',';
#endif
    {
        Fnf f(LOCALE_ru_RU_UTF_8, 1);
        assert(f.decimal_point() == sep);
    }
    {
        Fnt f(LOCALE_ru_RU_UTF_8, 1);
        assert(f.decimal_point() == sep);
    }
    {
        Fwf f(LOCALE_ru_RU_UTF_8, 1);
        assert(f.decimal_point() == wsep);
    }
    {
        Fwt f(LOCALE_ru_RU_UTF_8, 1);
        assert(f.decimal_point() == wsep);
    }

    {
        Fnf f(LOCALE_zh_CN_UTF_8, 1);
        assert(f.decimal_point() == '.');
    }
    {
        Fnt f(LOCALE_zh_CN_UTF_8, 1);
        assert(f.decimal_point() == '.');
    }
    {
        Fwf f(LOCALE_zh_CN_UTF_8, 1);
        assert(f.decimal_point() == L'.');
    }
    {
        Fwt f(LOCALE_zh_CN_UTF_8, 1);
        assert(f.decimal_point() == L'.');
    }

  return 0;
}
