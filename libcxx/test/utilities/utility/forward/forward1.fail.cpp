//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test forward

#include <utility>

struct A
{
};

A source() {return A();}
const A csource() {return A();}

int main()
{
    std::forward<A&>(source());  // error
}
