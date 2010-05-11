//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <mutex>

// class mutex;

// mutex(const mutex&) = delete;

#include <mutex>

int main()
{
    std::mutex m0;
    std::mutex m1(m0);
}
