//===-- asan_test_config.h ------------*- C++ -*-===//
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
#ifndef ASAN_TEST_CONFIG_H
#define ASAN_TEST_CONFIG_H

#include <vector>
#include <string>
#include <map>

#include "gtest/gtest.h"

using std::string;
using std::vector;
using std::map;

#ifndef ASAN_UAR
# error "please define ASAN_UAR"
#endif

#ifndef ASAN_HAS_EXCEPTIONS
# error "please define ASAN_HAS_EXCEPTIONS"
#endif

#ifndef ASAN_HAS_BLACKLIST
# error "please define ASAN_HAS_BLACKLIST"
#endif

#ifndef ASAN_NEEDS_SEGV
# error "please define ASAN_NEEDS_SEGV"
#endif

#ifndef ASAN_LOW_MEMORY
#define ASAN_LOW_MEMORY 0
#endif

#define ASAN_PCRE_DOTALL ""

#endif  // ASAN_TEST_CONFIG_H
