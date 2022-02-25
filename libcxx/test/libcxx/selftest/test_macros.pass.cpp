//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test the "test_macros.h" header.
#include <__config>
#include "test_macros.h"

#ifndef TEST_STD_VER
#error TEST_STD_VER must be defined
#endif

#ifndef TEST_NOEXCEPT
#error TEST_NOEXCEPT must be defined
#endif

#ifndef LIBCPP_ASSERT
#error LIBCPP_ASSERT must be defined
#endif

#ifndef LIBCPP_STATIC_ASSERT
#error LIBCPP_STATIC_ASSERT must be defined
#endif

void test_noexcept() TEST_NOEXCEPT
{
}

void test_libcxx_macros()
{
//  ===== C++14 features =====

//  defined(TEST_HAS_VARIABLE_TEMPLATES) != defined(_LIBCPP_HAS_NO_VARIABLE_TEMPLATES)
#ifdef TEST_HAS_VARIABLE_TEMPLATES
# ifdef _LIBCPP_HAS_NO_VARIABLE_TEMPLATES
#  error "TEST_VARIABLE_TEMPLATES mismatch (1)"
# endif
#else
# ifndef _LIBCPP_HAS_NO_VARIABLE_TEMPLATES
#  error "TEST_VARIABLE_TEMPLATES mismatch (2)"
# endif
#endif

//  ===== C++17 features =====
}

int main(int, char**)
{
    test_noexcept();
    test_libcxx_macros();

  return 0;
}
