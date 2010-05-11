//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <chrono>

// time_point

// Duration shall be an instance of duration.

#include <chrono>

int main()
{
    typedef std::chrono::time_point<std::chrono::system_clock, int> T;
    T t;
}
