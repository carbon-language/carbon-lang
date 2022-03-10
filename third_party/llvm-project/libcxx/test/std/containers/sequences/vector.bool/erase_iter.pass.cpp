//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>
// vector<bool>

// iterator erase(const_iterator position);

#include <vector>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    bool a1[] = {1, 0, 1};
    {
    std::vector<bool> l1(a1, a1+3);
    std::vector<bool>::const_iterator i = l1.begin();
    ++i;
    std::vector<bool>::iterator j = l1.erase(i);
    assert(l1.size() == 2);
    assert(std::distance(l1.begin(), l1.end()) == 2);
    assert(*j == true);
    assert(*l1.begin() == 1);
    assert(*std::next(l1.begin()) == true);
    j = l1.erase(j);
    assert(j == l1.end());
    assert(l1.size() == 1);
    assert(std::distance(l1.begin(), l1.end()) == 1);
    assert(*l1.begin() == true);
    j = l1.erase(l1.begin());
    assert(j == l1.end());
    assert(l1.size() == 0);
    assert(std::distance(l1.begin(), l1.end()) == 0);
    }
#if TEST_STD_VER >= 11
    {
    std::vector<bool, min_allocator<bool>> l1(a1, a1+3);
    std::vector<bool, min_allocator<bool>>::const_iterator i = l1.begin();
    ++i;
    std::vector<bool, min_allocator<bool>>::iterator j = l1.erase(i);
    assert(l1.size() == 2);
    assert(std::distance(l1.begin(), l1.end()) == 2);
    assert(*j == true);
    assert(*l1.begin() == 1);
    assert(*std::next(l1.begin()) == true);
    j = l1.erase(j);
    assert(j == l1.end());
    assert(l1.size() == 1);
    assert(std::distance(l1.begin(), l1.end()) == 1);
    assert(*l1.begin() == true);
    j = l1.erase(l1.begin());
    assert(j == l1.end());
    assert(l1.size() == 0);
    assert(std::distance(l1.begin(), l1.end()) == 0);
    }
#endif

  return 0;
}
