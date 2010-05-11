//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <thread>

// class thread

// unsigned hardware_concurrency();

#include <thread>
#include <cassert>

int main()
{
    assert(std::thread::hardware_concurrency() > 0);
}
