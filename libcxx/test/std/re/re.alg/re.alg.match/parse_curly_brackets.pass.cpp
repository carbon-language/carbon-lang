//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// template <class BidirectionalIterator, class Allocator, class charT, class traits>
//     bool
//     regex_match(BidirectionalIterator first, BidirectionalIterator last,
//                  match_results<BidirectionalIterator, Allocator>& m,
//                  const basic_regex<charT, traits>& e,
//                  regex_constants::match_flag_type flags = regex_constants::match_default);

// https://llvm.org/PR16135

#include <string>
#include <regex>
#include <cassert>
#include "test_macros.h"

void
test1()
{
    std::string re("\\{a\\}");
    std::string target("{a}");
    std::regex regex(re);
    std::smatch smatch;
    assert((std::regex_match(target, smatch, regex)));
}

void
test2()
{
    std::string re("\\{a\\}");
    std::string target("{a}");
    std::regex regex(re, std::regex::extended);
    std::smatch smatch;
    assert((std::regex_match(target, smatch, regex)));
}

void
test3()
{
    std::string re("\\{a\\}");
    std::string target("{a}");
    std::regex regex(re, std::regex::awk);
    std::smatch smatch;
    assert((std::regex_match(target, smatch, regex)));
}

void
test4()
{
    std::string re("\\{a\\}");
    std::string target("{a}");
    std::regex regex(re, std::regex::egrep);
    std::smatch smatch;
    assert((std::regex_match(target, smatch, regex)));
}

int main(int, char**)
{
    test1();
    test2();
    test3();
    test4();

  return 0;
}
