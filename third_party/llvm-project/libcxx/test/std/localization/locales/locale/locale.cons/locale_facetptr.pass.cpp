//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: locale.ru_RU.UTF-8

// This test relies on https://wg21.link/P0482 being implemented, which isn't in
// older Apple dylibs
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx{{10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0}}

// This test runs in C++20, but we have deprecated codecvt<char(16|32), char, mbstate_t> in C++20.
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <locale>

// template <class Facet> locale(const locale& other, Facet* f);

#include <locale>
#include <new>
#include <cassert>

#include "count_new.h"
#include "test_macros.h"
#include "platform_support.h" // locale name macros


template<class CharT>
void check_for(const std::locale& loc)
{
    assert(std::has_facet<std::collate<CharT> >(loc));

    assert(std::has_facet<std::ctype<CharT> >(loc));

    assert((std::has_facet<std::codecvt<CharT, char, std::mbstate_t> >(loc)));

    assert(std::has_facet<std::moneypunct<CharT> >(loc));
    assert(std::has_facet<std::money_get<CharT> >(loc));
    assert(std::has_facet<std::money_put<CharT> >(loc));

    assert(std::has_facet<std::numpunct<CharT> >(loc));
    assert(std::has_facet<std::num_get<CharT> >(loc));
    assert(std::has_facet<std::num_put<CharT> >(loc));

    assert(std::has_facet<std::time_get<CharT> >(loc));
    assert(std::has_facet<std::time_put<CharT> >(loc));

    assert(std::has_facet<std::messages<CharT> >(loc));
}

void check(const std::locale& loc)
{
    check_for<char>(loc);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    check_for<wchar_t>(loc);
#endif

    assert((std::has_facet<std::codecvt<char16_t, char, std::mbstate_t> >(loc)));
    assert((std::has_facet<std::codecvt<char32_t, char, std::mbstate_t> >(loc)));
#if TEST_STD_VER > 17
    assert((std::has_facet<std::codecvt<char16_t, char8_t, std::mbstate_t> >(loc)));
    assert((std::has_facet<std::codecvt<char32_t, char8_t, std::mbstate_t> >(loc)));
#endif
}

struct my_facet
    : public std::locale::facet
{
    int test() const {return 5;}

    static std::locale::id id;
};

std::locale::id my_facet::id;

int main(int, char**)
{
    {
        std::locale loc(LOCALE_ru_RU_UTF_8);
        check(loc);
        std::locale loc2(loc, new my_facet);
        check(loc2);
        assert((std::has_facet<my_facet>(loc2)));
        const my_facet& f = std::use_facet<my_facet>(loc2);
        assert(f.test() == 5);
    }
    assert(globalMemCounter.checkOutstandingNewEq(0));
    {
        std::locale loc;
        check(loc);
        std::locale loc2(loc, (std::ctype<char>*)0);
        check(loc2);
        assert(loc == loc2);
    }
    assert(globalMemCounter.checkOutstandingNewEq(0));

  return 0;
}
