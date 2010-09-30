//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <atomic>

// struct atomic_flag

// atomic_flag() = ATOMIC_FLAG_INIT;

#include <atomic>
#include <cassert>

int main()
{
    std::atomic_flag f = ATOMIC_FLAG_INIT;
    assert(f.test_and_set() == 0);
}
