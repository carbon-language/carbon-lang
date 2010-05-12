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
// class normal_distribution;

// void param(const param_type& parm);

#include <random>
#include <cassert>

int main()
{
    {
        typedef std::normal_distribution<> D;
        typedef D::param_type P;
        P p(0.25, 5.5);
        D d(0.75, 4);
        d.param(p);
        assert(d.param() == p);
    }
}
