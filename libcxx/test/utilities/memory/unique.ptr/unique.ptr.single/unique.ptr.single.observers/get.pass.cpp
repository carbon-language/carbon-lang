//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// test get

#include <memory>
#include <cassert>

int main()
{
    int* p = new int;
    std::unique_ptr<int> s(p);
    assert(s.get() == p);
}
