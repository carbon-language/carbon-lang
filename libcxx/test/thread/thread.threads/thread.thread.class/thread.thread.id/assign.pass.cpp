//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <thread>

// class thread::id

// id& operator=(const id&) = default;

#include <thread>
#include <cassert>

int main()
{
    std::thread::id id0;
    std::thread::id id1;
    id1 = id0;
    assert(id1 == id0);
    id1 = std::this_thread::get_id();
    assert(id1 != id0);
}
