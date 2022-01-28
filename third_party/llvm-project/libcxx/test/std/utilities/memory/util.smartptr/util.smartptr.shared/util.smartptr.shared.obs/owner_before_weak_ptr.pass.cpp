//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// template <class U> bool owner_before(weak_ptr<U> const& b) const noexcept;

#include <memory>
#include <cassert>
#include "test_macros.h"

int main(int, char**)
{
    const std::shared_ptr<int> p1(new int);
    const std::shared_ptr<int> p2 = p1;
    const std::shared_ptr<int> p3(new int);
    const std::weak_ptr<int> w1(p1);
    const std::weak_ptr<int> w2(p2);
    const std::weak_ptr<int> w3(p3);
    assert(!p1.owner_before(w2));
    assert(!p2.owner_before(w1));
    assert(p1.owner_before(w3) || p3.owner_before(w1));
    assert(p3.owner_before(w1) == p3.owner_before(w2));
    ASSERT_NOEXCEPT(p1.owner_before(w2));

  return 0;
}
