//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> class ctype;

// explicit ctype(size_t refs = 0);

// XFAIL: no-wide-characters

#include <locale>
#include <cassert>

#include "test_macros.h"

template <class C>
class my_facet
    : public std::ctype<C>
{
public:
    static int count;

    explicit my_facet(std::size_t refs = 0)
        : std::ctype<C>(refs) {++count;}

    ~my_facet() {--count;}
};

template <class C> int my_facet<C>::count = 0;

int main(int, char**)
{
    {
        std::locale l(std::locale::classic(), new my_facet<wchar_t>);
        assert(my_facet<wchar_t>::count == 1);
    }
    assert(my_facet<wchar_t>::count == 0);
    {
        my_facet<wchar_t> f(1);
        assert(my_facet<wchar_t>::count == 1);
        {
            std::locale l(std::locale::classic(), &f);
            assert(my_facet<wchar_t>::count == 1);
        }
        assert(my_facet<wchar_t>::count == 1);
    }
    assert(my_facet<wchar_t>::count == 0);

  return 0;
}
