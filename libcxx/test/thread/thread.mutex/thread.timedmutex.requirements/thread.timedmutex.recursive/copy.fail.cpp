//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <mutex>

// class recursive_timed_mutex;

// recursive_timed_mutex(const recursive_timed_mutex&) = delete;

#include <mutex>

int main()
{
    std::recursive_timed_mutex m0;
    std::recursive_timed_mutex m1(m0);
}
