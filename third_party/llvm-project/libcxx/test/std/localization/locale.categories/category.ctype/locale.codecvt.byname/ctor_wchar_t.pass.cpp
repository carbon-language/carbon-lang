//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: locale.en_US.UTF-8

// <locale>

// template <> class codecvt_byname<wchar_t, char, mbstate_t>

// explicit codecvt_byname(const char* nm, size_t refs = 0);
// explicit codecvt_byname(const string& nm, size_t refs = 0);

// XFAIL: no-wide-characters

#include <locale>
#include <cassert>

#include "test_macros.h"
#include "platform_support.h" // locale name macros

typedef std::codecvt_byname<wchar_t, char, std::mbstate_t> F;

class my_facet
    : public F
{
public:
    static int count;

    explicit my_facet(const char* nm, std::size_t refs = 0)
        : F(nm, refs) {++count;}
    explicit my_facet(const std::string& nm, std::size_t refs = 0)
        : F(nm, refs) {++count;}

    ~my_facet() {--count;}
};

int my_facet::count = 0;

int main(int, char**)
{
    {
        std::locale l(std::locale::classic(), new my_facet(LOCALE_en_US_UTF_8));
        assert(my_facet::count == 1);
    }
    assert(my_facet::count == 0);
    {
        my_facet f(LOCALE_en_US_UTF_8, 1);
        assert(my_facet::count == 1);
        {
            std::locale l(std::locale::classic(), &f);
            assert(my_facet::count == 1);
        }
        assert(my_facet::count == 1);
    }
    assert(my_facet::count == 0);
    {
        std::locale l(std::locale::classic(), new my_facet(std::string(LOCALE_en_US_UTF_8)));
        assert(my_facet::count == 1);
    }
    assert(my_facet::count == 0);
    {
        my_facet f(std::string(LOCALE_en_US_UTF_8), 1);
        assert(my_facet::count == 1);
        {
            std::locale l(std::locale::classic(), &f);
            assert(my_facet::count == 1);
        }
        assert(my_facet::count == 1);
    }
    assert(my_facet::count == 0);

  return 0;
}
