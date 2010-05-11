//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
 
// test terminate_handler

#include <exception>

void f() {}

int main()
{
    std::terminate_handler p = f;
}
