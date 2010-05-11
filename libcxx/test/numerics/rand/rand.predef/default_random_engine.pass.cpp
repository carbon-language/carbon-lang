//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <random>

// typedef minstd_rand0 default_random_engine;



#include <random>
#include <cassert>

int main()
{
    std::default_random_engine e;
    e.discard(9999);
    assert(e() == 1043618065u);
}
