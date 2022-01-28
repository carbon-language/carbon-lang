//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_class

#include <type_traits>
#include <cstddef>        // for std::nullptr_t
#include "test_macros.h"

template <class T>
void test_is_class()
{
    static_assert( std::is_class<T>::value, "");
    static_assert( std::is_class<const T>::value, "");
    static_assert( std::is_class<volatile T>::value, "");
    static_assert( std::is_class<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert( std::is_class_v<T>, "");
    static_assert( std::is_class_v<const T>, "");
    static_assert( std::is_class_v<volatile T>, "");
    static_assert( std::is_class_v<const volatile T>, "");
#endif
}

template <class T>
void test_is_not_class()
{
    static_assert(!std::is_class<T>::value, "");
    static_assert(!std::is_class<const T>::value, "");
    static_assert(!std::is_class<volatile T>::value, "");
    static_assert(!std::is_class<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert(!std::is_class_v<T>, "");
    static_assert(!std::is_class_v<const T>, "");
    static_assert(!std::is_class_v<volatile T>, "");
    static_assert(!std::is_class_v<const volatile T>, "");
#endif
}

class Empty
{
};

class NotEmpty
{
    virtual ~NotEmpty();
};

union Union {};

struct bit_zero
{
    int :  0;
};

class Abstract
{
    virtual ~Abstract() = 0;
};

enum Enum {zero, one};
struct incomplete_type;

typedef void (*FunctionPtr)();

int main(int, char**)
{
    test_is_class<Empty>();
    test_is_class<bit_zero>();
    test_is_class<NotEmpty>();
    test_is_class<Abstract>();
    test_is_class<incomplete_type>();

#if TEST_STD_VER >= 11
// In C++03 we have an emulation of std::nullptr_t
    test_is_not_class<std::nullptr_t>();
#endif
    test_is_not_class<void>();
    test_is_not_class<int>();
    test_is_not_class<int&>();
#if TEST_STD_VER >= 11
    test_is_not_class<int&&>();
#endif
    test_is_not_class<int*>();
    test_is_not_class<double>();
    test_is_not_class<const int*>();
    test_is_not_class<char[3]>();
    test_is_not_class<char[]>();
    test_is_not_class<Enum>();
    test_is_not_class<Union>();
    test_is_not_class<FunctionPtr>();

  return 0;
}
