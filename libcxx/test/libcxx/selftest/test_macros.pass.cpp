//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Test the "test_macros.h" header.
#include "test_macros.h"

#ifndef TEST_STD_VER
#error TEST_STD_VER must be defined
#endif

#ifndef TEST_DECLTYPE
#error TEST_DECLTYPE must be defined
#endif

#ifndef TEST_NOEXCEPT
#error TEST_NOEXCEPT must be defined
#endif

#ifndef TEST_STATIC_ASSERT
#error TEST_STATIC_ASSERT must be defined
#endif

template <class T, class U>
struct is_same { enum { value = 0 }; };

template <class T>
struct is_same<T, T> { enum { value = 1 }; };

int foo() { return 0; }

void test_noexcept() TEST_NOEXCEPT
{
}

void test_decltype()
{
  typedef TEST_DECLTYPE(foo()) MyType;
  TEST_STATIC_ASSERT((is_same<MyType, int>::value), "is same");
}

void test_static_assert()
{
    TEST_STATIC_ASSERT((is_same<int, int>::value), "is same");
    TEST_STATIC_ASSERT((!is_same<int, long>::value), "not same");
}

int main()
{
    test_noexcept();
    test_decltype();
    test_static_assert();
}
