//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// template<class T, class... Args> shared_ptr<T> make_shared(Args&&... args);

#include <memory>
#include <cassert>

#include "test_macros.h"

template <typename T>
void test(const T &t0)
{
    {
    T t1 = t0;
    std::shared_ptr<T> p0 = std::make_shared<T>(t0);
    std::shared_ptr<T> p1 = std::make_shared<T>(t1);
    assert(*p0 == t0);
    assert(*p1 == t1);
    }

    {
    const T t1 = t0;
    std::shared_ptr<const T> p0 = std::make_shared<const T>(t0);
    std::shared_ptr<const T> p1 = std::make_shared<const T>(t1);
    assert(*p0 == t0);
    assert(*p1 == t1);
    }

    {
    volatile T t1 = t0;
    std::shared_ptr<volatile T> p0 = std::make_shared<volatile T>(t0);
    std::shared_ptr<volatile T> p1 = std::make_shared<volatile T>(t1);
    assert(*p0 == t0);
    assert(*p1 == t1);
    }

    {
    const volatile T t1 = t0;
    std::shared_ptr<const volatile T> p0 = std::make_shared<const volatile T>(t0);
    std::shared_ptr<const volatile T> p1 = std::make_shared<const volatile T>(t1);
    assert(*p0 == t0);
    assert(*p1 == t1);
    }

}

int main(int, char**)
{
    test<bool>(true);
    test<int>(3);
    test<double>(5.0);

  return 0;
}
