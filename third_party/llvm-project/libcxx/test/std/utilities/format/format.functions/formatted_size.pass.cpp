//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-format
// TODO FMT Evaluate gcc-11 status
// UNSUPPORTED: gcc-11
// TODO FMT Investigate AppleClang ICE
// UNSUPPORTED: apple-clang-13

// <format>

// template<class... Args>
//   size_t formatted_size(format-string<Args...> fmt, const Args&... args);
// template<class... Args>
//   size_t formatted_size(wformat-string<Args...> fmt, const Args&... args);

#include <format>
#include <cassert>
#include <vector>

#include "test_macros.h"
#include "format_tests.h"
#include "string_literal.h"

auto test = []<string_literal fmt, class CharT, class... Args>(std::basic_string_view<CharT> expected,
                                                               const Args&... args) constexpr {
  size_t size = std::formatted_size(fmt.template sv<CharT>(), args...);
  assert(size == expected.size());
};

auto test_exception = []<class CharT, class... Args>(std::string_view, std::basic_string_view<CharT>, const Args&...) {
  // After P2216 most exceptions thrown by std::formatted_siz3 become ill-formed.
  // Therefore this tests does nothing.
  // A basic ill-formed test is done in formatted_size.verify.cpp
  // The exceptions are tested by other functions that don't use the basic-format-string as fmt argument.
};

int main(int, char**) {
  format_tests<char>(test, test_exception);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  format_tests_char_to_wchar_t(test);
  format_tests<wchar_t>(test, test_exception);
#endif

  return 0;
}
