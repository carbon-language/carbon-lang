//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
 
// test set_new_handler

#include <new>
#include <cassert>

void f1() {}
void f2() {}

int main()
{
    assert(std::set_new_handler(f1) == 0);
    assert(std::set_new_handler(f2) == f1);
}
