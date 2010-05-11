//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration_values::min

#include <chrono>
#include <limits>
#include <cassert>

#include "../../rep.h"

int main()
{
    assert(std::chrono::duration_values<int>::min() ==
           std::numeric_limits<int>::lowest());
    assert(std::chrono::duration_values<double>::min() ==
           std::numeric_limits<double>::lowest());
    assert(std::chrono::duration_values<Rep>::min() ==
           std::numeric_limits<Rep>::lowest());
}
