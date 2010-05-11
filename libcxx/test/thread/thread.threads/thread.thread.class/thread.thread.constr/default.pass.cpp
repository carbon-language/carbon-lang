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

// thread();

#include <thread>
#include <cassert>

int main()
{
    std::thread t;
    assert(t.get_id() == std::thread::id());
}
