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

// test reset against resetting self

#include <memory>

struct A
{
    std::unique_ptr<A> ptr_;

    A() : ptr_(this) {}
    void reset() {ptr_.reset();}
};

int main()
{
    (new A)->reset();
}
