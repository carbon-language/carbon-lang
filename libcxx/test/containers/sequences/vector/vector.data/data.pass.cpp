//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>

// pointer data();

#include <vector>
#include <cassert>

#include "../../../min_allocator.h"

int main()
{
    {
        std::vector<int> v;
        assert(v.data() == 0);
    }
    {
        std::vector<int> v(100);
        assert(v.data() == &v.front());
    }
#if __cplusplus >= 201103L
    {
        std::vector<int, min_allocator<int>> v;
        assert(v.data() == 0);
    }
    {
        std::vector<int, min_allocator<int>> v(100);
        assert(v.data() == &v.front());
    }
#endif
}
