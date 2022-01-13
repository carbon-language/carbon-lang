//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// class ctype_base
// {
// public:
//     typedef T mask;
//
//     // numeric values are for exposition only.
//     static const mask space = 1 << 0;
//     static const mask print = 1 << 1;
//     static const mask cntrl = 1 << 2;
//     static const mask upper = 1 << 3;
//     static const mask lower = 1 << 4;
//     static const mask alpha = 1 << 5;
//     static const mask digit = 1 << 6;
//     static const mask punct = 1 << 7;
//     static const mask xdigit = 1 << 8;
//     static const mask alnum = alpha | digit;
//     static const mask graph = alnum | punct;
// };

#include <locale>
#include <cassert>

#include "test_macros.h"

template <class T>
void test(const T &) {}

int main(int, char**)
{
    assert(std::ctype_base::space);
    assert(std::ctype_base::print);
    assert(std::ctype_base::cntrl);
    assert(std::ctype_base::upper);
    assert(std::ctype_base::lower);
    assert(std::ctype_base::alpha);
    assert(std::ctype_base::digit);
    assert(std::ctype_base::punct);
    assert(std::ctype_base::xdigit);
    assert(
      ( std::ctype_base::space
      & std::ctype_base::print
      & std::ctype_base::cntrl
      & std::ctype_base::upper
      & std::ctype_base::lower
      & std::ctype_base::alpha
      & std::ctype_base::digit
      & std::ctype_base::punct
      & std::ctype_base::xdigit) == 0);
    assert(std::ctype_base::alnum == (std::ctype_base::alpha | std::ctype_base::digit));
    assert(std::ctype_base::graph == (std::ctype_base::alnum | std::ctype_base::punct));

    test(std::ctype_base::space);
    test(std::ctype_base::print);
    test(std::ctype_base::cntrl);
    test(std::ctype_base::upper);
    test(std::ctype_base::lower);
    test(std::ctype_base::alpha);
    test(std::ctype_base::digit);
    test(std::ctype_base::punct);
    test(std::ctype_base::xdigit);
    test(std::ctype_base::blank);
    test(std::ctype_base::alnum);
    test(std::ctype_base::graph);

  return 0;
}
