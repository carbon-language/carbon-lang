//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <mutex>

// struct once_flag;

// once_flag(const once_flag&) = delete;

#include <mutex>

int main()
{
    std::once_flag f;
    std::once_flag f2(f);
}
