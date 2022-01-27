//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test_macros.h"

#if TEST_STD_VER < 11
#error
#else

// <iterator>
// template <class C> auto begin(C& c) -> decltype(c.begin());
// template <class C> auto begin(const C& c) -> decltype(c.begin());
// template <class C> auto end(C& c) -> decltype(c.end());
// template <class C> auto end(const C& c) -> decltype(c.end());
// template <class E> reverse_iterator<const E*> rbegin(initializer_list<E> il);
// template <class E> reverse_iterator<const E*> rend(initializer_list<E> il);


#include <iterator>
#include <cassert>

namespace Foo {
    struct FakeContainer {};
    typedef int FakeIter;

    FakeIter begin(const FakeContainer &)   { return 1; }
    FakeIter end  (const FakeContainer &)   { return 2; }
    FakeIter rbegin(const FakeContainer &)  { return 3; }
    FakeIter rend  (const FakeContainer &)  { return 4; }

    FakeIter cbegin(const FakeContainer &)  { return 11; }
    FakeIter cend  (const FakeContainer &)  { return 12; }
    FakeIter crbegin(const FakeContainer &) { return 13; }
    FakeIter crend  (const FakeContainer &) { return 14; }
}


int main(int, char**) {
// Bug #28927 - shouldn't find these via ADL
    TEST_IGNORE_NODISCARD  std::cbegin (Foo::FakeContainer());
    TEST_IGNORE_NODISCARD  std::cend   (Foo::FakeContainer());
    TEST_IGNORE_NODISCARD  std::crbegin(Foo::FakeContainer());
    TEST_IGNORE_NODISCARD  std::crend  (Foo::FakeContainer());

  return 0;
}
#endif
