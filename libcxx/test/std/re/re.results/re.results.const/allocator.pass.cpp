//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// class match_results<BidirectionalIterator, Allocator>

// match_results(const Allocator& a = Allocator());

#include <regex>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"

template <class CharT, class Allocator>
void
test(const Allocator& a)
{
    std::match_results<const CharT*, Allocator> m(a);
    assert(m.size() == 0);
    assert(!m.ready());
    assert(m.get_allocator() == a);
}

int main(int, char**)
{
    test<char>(test_allocator<std::sub_match<const char*> >(3));
    test<wchar_t>(test_allocator<std::sub_match<const wchar_t*> >(3));

  return 0;
}
