//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class gamma_distribution

// result_type max() const;

#include <random>
#include <cassert>

int main()
{
    {
        typedef std::gamma_distribution<> D;
        D d(5, .25);
        D::result_type m = d.max();
        assert(m == INFINITY);
    }
}
