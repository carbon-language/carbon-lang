//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <chrono>

// time_point

// Test default template arg:

// template <class Clock, class Duration = typename Clock::duration> 
//   class time_point;

#include <chrono>
#include <type_traits>

int main()
{
    static_assert((std::is_same<std::chrono::system_clock::duration,
                   std::chrono::time_point<std::chrono::system_clock>::duration>::value), "");
}
