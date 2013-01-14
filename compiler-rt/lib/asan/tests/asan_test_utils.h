//===-- asan_test_utils.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
//===----------------------------------------------------------------------===//

#ifndef ASAN_TEST_UTILS_H
#define ASAN_TEST_UTILS_H

#if !defined(ASAN_EXTERNAL_TEST_CONFIG)
# define INCLUDED_FROM_ASAN_TEST_UTILS_H
# include "asan_test_config.h"
# undef INCLUDED_FROM_ASAN_TEST_UTILS_H
#endif

#include "sanitizer_common/tests/sanitizer_test_utils.h"

// Check that pthread_create/pthread_join return success.
#define PTHREAD_CREATE(a, b, c, d) ASSERT_EQ(0, pthread_create(a, b, c, d))
#define PTHREAD_JOIN(a, b) ASSERT_EQ(0, pthread_join(a, b))

#endif  // ASAN_TEST_UTILS_H
