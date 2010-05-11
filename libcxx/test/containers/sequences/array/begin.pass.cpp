//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <array>

// iterator begin();

#include <array>
#include <cassert>

int main()
{
    {
        typedef double T;
        typedef std::array<T, 3> C;
        C c = {1, 2, 3.5};
        C::iterator i = c.begin();
        assert(*i == 1);
        assert(&*i == c.data());
        *i = 5.5;
        assert(c[0] == 5.5);
    }
    {
    }
}
