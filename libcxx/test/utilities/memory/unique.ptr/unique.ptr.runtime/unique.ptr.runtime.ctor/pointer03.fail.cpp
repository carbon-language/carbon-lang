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

// Test unique_ptr<T[]>(pointer) ctor

// unique_ptr<T[]>(pointer) ctor should require non-pointer Deleter

#include <memory>

int main()
{
    std::unique_ptr<int[], void (*)(void*)> p(new int);
}
