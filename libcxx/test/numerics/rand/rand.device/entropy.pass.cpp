//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <random>

// class random_device;

// double entropy() const;

#include <random>
#include <cassert>


int main()
{
    std::random_device r;
    double e = r.entropy();
}
