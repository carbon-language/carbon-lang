//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <mutex>

// class timed_mutex;

// timed_mutex& operator=(const timed_mutex&) = delete;

#include <mutex>

int main()
{
    std::timed_mutex m0;
    std::timed_mutex m1;
    m1 = m0;
}
