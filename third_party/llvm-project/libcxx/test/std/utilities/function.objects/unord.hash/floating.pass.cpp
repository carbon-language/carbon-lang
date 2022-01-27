//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <functional>

// template <class T>
// struct hash
//     : public unary_function<T, size_t>
// {
//     size_t operator()(T val) const;
// };

// Not very portable

#include <functional>
#include <cassert>
#include <type_traits>
#include <limits>
#include <cmath>

#include "test_macros.h"

template <class T>
void
test()
{
    typedef std::hash<T> H;
#if TEST_STD_VER <= 17
    static_assert((std::is_same<typename H::argument_type, T>::value), "");
    static_assert((std::is_same<typename H::result_type, std::size_t>::value), "");
#endif
    ASSERT_NOEXCEPT(H()(T()));
    H h;

    std::size_t t0 = h(0.);
    std::size_t tn0 = h(-0.);
    std::size_t tp1 = h(static_cast<T>(0.1));
    std::size_t t1 = h(1);
    std::size_t tn1 = h(-1);
    std::size_t pinf = h(INFINITY);
    std::size_t ninf = h(-INFINITY);
    assert(t0 == tn0);
    assert(t0 != tp1);
    assert(t0 != t1);
    assert(t0 != tn1);
    assert(t0 != pinf);
    assert(t0 != ninf);

    assert(tp1 != t1);
    assert(tp1 != tn1);
    assert(tp1 != pinf);
    assert(tp1 != ninf);

    assert(t1 != tn1);
    assert(t1 != pinf);
    assert(t1 != ninf);

    assert(tn1 != pinf);
    assert(tn1 != ninf);

    assert(pinf != ninf);
}

int main(int, char**)
{
    test<float>();
    test<double>();
    test<long double>();

  return 0;
}
