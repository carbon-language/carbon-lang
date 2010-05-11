//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <array>

// array();

#include <array>
#include <cassert>

int main()
{
    {
        typedef double T;
        typedef std::array<T, 3> C;
        C c;
        assert(c.size() == 3);
    }
    {
        typedef double T;
        typedef std::array<T, 0> C;
        C c;
        assert(c.size() == 0);
    }
}
