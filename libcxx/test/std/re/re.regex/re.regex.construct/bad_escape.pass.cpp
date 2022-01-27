//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions
// <regex>

// template <class charT, class traits = regex_traits<charT>> class basic_regex;

// template <class ST, class SA>
//    basic_regex(const basic_string<charT, ST, SA>& s);

#include <regex>
#include <cassert>
#include "test_macros.h"

static bool error_escape_thrown(const char *pat)
{
    bool result = false;
    try {
        std::regex re(pat);
    } catch (const std::regex_error &ex) {
        result = (ex.code() == std::regex_constants::error_escape);
    }
    return result;
}

int main(int, char**)
{
    assert(error_escape_thrown("[\\a]"));
    assert(error_escape_thrown("\\a"));
    assert(error_escape_thrown("\\"));

    assert(error_escape_thrown("[\\e]"));
    assert(error_escape_thrown("\\e"));

    assert(error_escape_thrown("[\\c:]"));
    assert(error_escape_thrown("\\c:"));
    assert(error_escape_thrown("\\c"));
    assert(!error_escape_thrown("[\\cA]"));
    assert(!error_escape_thrown("\\cA"));


  return 0;
}
