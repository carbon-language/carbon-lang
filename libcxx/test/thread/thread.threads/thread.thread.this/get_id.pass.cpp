//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <thread>

// thread::id this_thread::get_id();

#include <thread>
#include <cassert>

int main()
{
    std::thread::id id = std::this_thread::get_id();
    assert(id != std::thread::id());
}
