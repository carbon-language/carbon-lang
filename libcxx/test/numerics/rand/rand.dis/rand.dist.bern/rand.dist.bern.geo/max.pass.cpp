//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <random>

// template<class IntType = int>
// class geometric_distribution

// result_type max() const;

#include <random>
#include <cassert>

int main()
{
    {
        typedef std::geometric_distribution<> D;
        D d(.25);
        assert(d.max() == std::numeric_limits<int>::max());
    }
}
