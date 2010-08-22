//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// struct allocator_arg_t { };
// const allocator_arg_t allocator_arg = allocator_arg_t();

#include <memory>

void test(std::allocator_arg_t) {}

int main()
{
    test(std::allocator_arg);
}
