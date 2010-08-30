//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <future>

// class packaged_task<R(ArgTypes...)>

// packaged_task();

#include <future>
#include <cassert>

struct A {};

int main()
{
    std::packaged_task<A(int, char)> p;
    assert(!p);
}
