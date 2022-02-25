//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <string>

// Test that <string> provides all of the arithmetic, enum, and pointer
// hash specializations.

#include <string>

#include "poisoned_hash_helper.h"

#include "test_macros.h"

int main(int, char**) {
  test_library_hash_specializations_available();
  {
    test_hash_enabled_for_type<std::string>();
    test_hash_enabled_for_type<std::wstring>();
#if defined(__cpp_lib_char8_t) && __cpp_lib_char8_t >= 201811L
    test_hash_enabled_for_type<std::u8string>();
#endif
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
    test_hash_enabled_for_type<std::u16string>();
    test_hash_enabled_for_type<std::u32string>();
#endif
  }

  return 0;
}
