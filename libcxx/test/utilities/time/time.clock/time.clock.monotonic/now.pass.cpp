//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <chrono>

// monotonic_clock

// static time_point now();

#include <chrono>
#include <cassert>

int main()
{
    typedef std::chrono::monotonic_clock C;
    C::time_point t1 = C::now();
    C::time_point t2 = C::now();
    assert(t2 >= t1);
}
