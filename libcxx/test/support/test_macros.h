// -*- C++ -*-
//===---------------------------- test_macros.h ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TEST_MACROS_HPP
#define SUPPORT_TEST_MACROS_HPP

#define TEST_CONCAT1(X, Y) X##Y
#define TEST_CONCAT(X, Y) TEST_CONCAT1(X, Y)

#ifdef __has_extension
#define TEST_HAS_EXTENSION(X) __has_extension(X)
#else
#define TEST_HAS_EXTENSION(X) 0
#endif

/* Make a nice name for the standard version */
#if  __cplusplus <= 199711L
# define TEST_STD_VER 3
#elif __cplusplus <= 201103L
# define TEST_STD_VER 11
#elif __cplusplus <= 201402L
# define TEST_STD_VER 14
#else
# define TEST_STD_VER 99    // greater than current standard
#endif

/* Features that were introduced in C++11 */
#if TEST_STD_VER >= 11
#define TEST_HAS_RVALUE_REFERENCES
#define TEST_HAS_VARIADIC_TEMPLATES
#define TEST_HAS_INITIALIZER_LISTS
#define TEST_HAS_BASIC_CONSTEXPR
#endif

/* Features that were introduced in C++14 */
#if TEST_STD_VER >= 14
#define TEST_HAS_EXTENDED_CONSTEXPR
#define TEST_HAS_VARIABLE_TEMPLATES
#endif

/* Features that were introduced after C++14 */
#if TEST_STD_VER > 14
#endif

#if TEST_HAS_EXTENSION(cxx_decltype) || TEST_STD_VER >= 11
#define TEST_DECLTYPE(T) decltype(T)
#else
#define TEST_DECLTYPE(T) __typeof__(T)
#endif

#if TEST_STD_VER >= 11
#define TEST_NOEXCEPT noexcept
#else
#define TEST_NOEXCEPT
#endif

#if TEST_HAS_EXTENSION(cxx_static_assert) || TEST_STD_VER >= 11
#  define TEST_STATIC_ASSERT(Expr, Msg) static_assert(Expr, Msg)
#else
#  define TEST_STATIC_ASSERT(Expr, Msg)                          \
      typedef ::test_detail::static_assert_check<sizeof(         \
          ::test_detail::static_assert_incomplete_test<(Expr)>)> \
    TEST_CONCAT(test_assert, __LINE__)
#
#endif

namespace test_detail {

template <bool> struct static_assert_incomplete_test;
template <> struct static_assert_incomplete_test<true> {};
template <unsigned> struct static_assert_check {};

} // end namespace test_detail


#endif // SUPPORT_TEST_MACROS_HPP
