//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <thread>

// void this_thread::yield();

#include <thread>
#include <cassert>

int main()
{
    std::this_thread::yield();
}
