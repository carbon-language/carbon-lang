//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// reference_wrapper

// Test that reference wrapper meets the requirements of CopyConstructible and
// CopyAssignable, and TriviallyCopyable (starting in C++14).

#include <functional>
#include <type_traits>
#include <string>

#include "test_macros.h"

#if TEST_STD_VER >= 11
class MoveOnly
{
    MoveOnly(const MoveOnly&);
    MoveOnly& operator=(const MoveOnly&);

    int data_;
public:
    MoveOnly(int data = 1) : data_(data) {}
    MoveOnly(MoveOnly&& x)
        : data_(x.data_) {x.data_ = 0;}
    MoveOnly& operator=(MoveOnly&& x)
        {data_ = x.data_; x.data_ = 0; return *this;}

    int get() const {return data_;}
};
#endif


template <class T>
void test()
{
    typedef std::reference_wrapper<T> Wrap;
    static_assert(std::is_copy_constructible<Wrap>::value, "");
    static_assert(std::is_copy_assignable<Wrap>::value, "");
#if TEST_STD_VER >= 14
    static_assert(std::is_trivially_copyable<Wrap>::value, "");
#endif
}

int main(int, char**)
{
    test<int>();
    test<double>();
    test<std::string>();
#if TEST_STD_VER >= 11
    test<MoveOnly>();
#endif

  return 0;
}
