//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <chrono>

// system_clock

// time_t to_time_t(const time_point& t);

#include <chrono>
#include <ctime>

int main()
{
    typedef std::chrono::system_clock C;
    std::time_t t1 = C::to_time_t(C::now());
}
