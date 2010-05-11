//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <exception>

// typedef unspecified exception_ptr;

// exception_ptr shall satisfy the requirements of NullablePointer.

#include <exception>
#include <cassert>

int main()
{
    std::exception_ptr p;
    assert(p == nullptr);
    std::exception_ptr p2 = p;
    assert(nullptr == p);
    assert(!p);
    assert(p2 == p);
    p2 = p;
    assert(p2 == p);
    assert(p2 == nullptr);
    std::exception_ptr p3 = nullptr;
    assert(p3 == nullptr);
    p3 = nullptr;
    assert(p3 == nullptr);
}
