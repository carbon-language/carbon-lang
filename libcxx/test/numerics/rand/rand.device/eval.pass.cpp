//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <random>

// class random_device;

// result_type operator()();

#include <random>
#include <cassert>

int main()
{
    std::random_device r;
    std::random_device::result_type e = r();
}
