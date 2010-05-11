//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <thread>

// class thread::id

// id();

#include <thread>
#include <cassert>

int main()
{
    std::thread::id id;
    assert(id == std::thread::id());
}
