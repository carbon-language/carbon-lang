//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
 
// test set_unexpected

#include <exception>
#include <cassert>

void f1() {}
void f2() {}

int main()
{
    assert(std::set_unexpected(f1) == std::terminate);
    assert(std::set_unexpected(f2) == f1);
}
