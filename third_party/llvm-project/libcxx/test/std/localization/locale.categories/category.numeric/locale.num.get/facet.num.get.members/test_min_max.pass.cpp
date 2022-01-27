//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>

#include "test_macros.h"

template <class T>
bool check_stream_failed(std::string const& val) {
    std::istringstream ss(val);
    T result;
    return !(ss >> result);
}

template<typename T>
void check_limits()
{
    const bool is_unsigned = std::is_unsigned<T>::value;
    T minv = std::numeric_limits<T>::min();
    T maxv = std::numeric_limits<T>::max();

    std::ostringstream miniss, maxiss;
    assert(miniss << minv);
    assert(maxiss << maxv);
    std::string mins = miniss.str();
    std::string maxs = maxiss.str();

    std::istringstream maxoss(maxs), minoss(mins);

    T new_minv, new_maxv;
    assert(maxoss >> new_maxv);
    assert(minoss >> new_minv);

    assert(new_minv == minv);
    assert(new_maxv == maxv);

    maxs[maxs.size() - 1]++;
    assert(check_stream_failed<T>(maxs));
    if (!is_unsigned) {
        mins[mins.size() - 1]++;
        assert(check_stream_failed<T>(mins));
    }
}

int main(int, char**)
{
    check_limits<short>();
    check_limits<unsigned short>();
    check_limits<int>();
    check_limits<unsigned int>();
    check_limits<long>();
    check_limits<unsigned long>();
    check_limits<long long>();
    check_limits<unsigned long long>();

  return 0;
}
