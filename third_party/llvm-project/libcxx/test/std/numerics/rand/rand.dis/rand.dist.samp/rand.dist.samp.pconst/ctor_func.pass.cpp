//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class piecewise_constant_distribution

// template<class UnaryOperation>
//     piecewise_constant_distribution(size_t nw, result_type xmin,
//                                     result_type xmax, UnaryOperation fw);

#include <random>
#include <cassert>

#include "test_macros.h"

double fw(double x)
{
    return 2*x;
}

int main(int, char**)
{
    {
        typedef std::piecewise_constant_distribution<> D;
        D d(0, 0, 1, fw);
        std::vector<double> iv = d.intervals();
        assert(iv.size() == 2);
        assert(iv[0] == 0);
        assert(iv[1] == 1);
        std::vector<double> dn = d.densities();
        assert(dn.size() == 1);
        assert(dn[0] == 1);
    }
    {
        typedef std::piecewise_constant_distribution<> D;
        D d(1, 10, 12, fw);
        std::vector<double> iv = d.intervals();
        assert(iv.size() == 2);
        assert(iv[0] == 10);
        assert(iv[1] == 12);
        std::vector<double> dn = d.densities();
        assert(dn.size() == 1);
        assert(dn[0] == 0.5);
    }
    {
        typedef std::piecewise_constant_distribution<> D;
        D d(2, 6, 14, fw);
        std::vector<double> iv = d.intervals();
        assert(iv.size() == 3);
        assert(iv[0] == 6);
        assert(iv[1] == 10);
        assert(iv[2] == 14);
        std::vector<double> dn = d.densities();
        assert(dn.size() == 2);
        assert(dn[0] == 0.1);
        assert(dn[1] == 0.15);
    }

  return 0;
}
