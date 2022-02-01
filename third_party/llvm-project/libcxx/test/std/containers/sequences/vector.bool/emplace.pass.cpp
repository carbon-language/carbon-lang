//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <vector>
//  vector<bool>

// template <class... Args> iterator emplace(const_iterator pos, Args&&... args);

#include <vector>
#include <cassert>
#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        typedef std::vector<bool> C;
        C c;

        C::iterator i = c.emplace(c.cbegin());
        assert(i == c.begin());
        assert(c.size() == 1);
        assert(c.front() == false);

        i = c.emplace(c.cend(), true);
        assert(i == c.end()-1);
        assert(c.size() == 2);
        assert(c.front() == false);
        assert(c.back() == true);

        i = c.emplace(c.cbegin()+1, true);
        assert(i == c.begin()+1);
        assert(c.size() == 3);
        assert(c.front() == false);
        assert(c[1] == true);
        assert(c.back() == true);
    }
    {
        typedef std::vector<bool, min_allocator<bool>> C;
        C c;

        C::iterator i = c.emplace(c.cbegin());
        assert(i == c.begin());
        assert(c.size() == 1);
        assert(c.front() == false);

        i = c.emplace(c.cend(), true);
        assert(i == c.end()-1);
        assert(c.size() == 2);
        assert(c.front() == false);
        assert(c.back() == true);

        i = c.emplace(c.cbegin()+1, true);
        assert(i == c.begin()+1);
        assert(c.size() == 3);
        assert(c.size() == 3);
        assert(c.front() == false);
        assert(c[1] == true);
        assert(c.back() == true);
    }

  return 0;
}
