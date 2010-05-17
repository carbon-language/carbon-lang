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

// explicit geometric_distribution(const param_type& parm);

#include <random>
#include <cassert>

int main()
{
    {
        typedef std::geometric_distribution<> D;
        typedef D::param_type P;
        P p(0.25);
        D d(p);
        assert(d.p() == 0.25);
    }
}
