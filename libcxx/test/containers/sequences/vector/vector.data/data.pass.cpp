//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>

// pointer data();

#include <vector>
#include <cassert>

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
}
