//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <mutex>

// template <class Mutex> class unique_lock;

// unique_lock();

#include <mutex>
#include <cassert>

int main()
{
    std::unique_lock<std::mutex> ul;
    assert(!ul.owns_lock());
    assert(ul.mutex() == nullptr);
}
