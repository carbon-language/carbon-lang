//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization
// UNSUPPORTED: libcpp-has-no-incomplete-format
// TODO FMT Evaluate gcc-11 status
// UNSUPPORTED: gcc-11

// Basic test to validate ill-formed code is properly detected.

// <format>

// template<class Out, class... Args>
//   Out format_to(Out out, const locale& loc,
//                 format-string<Args...> fmt, const Args&... args);
// template<class Out, class... Args>
//   Out format_to(Out out, const locale& loc,
//                 wformat-string<Args...> fmt, const Args&... args);

#include <format>
#include <locale>

#include "test_macros.h"

extern char* out;
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
extern wchar_t* wout;
#endif

// clang-format off

void f() {
  std::format_to(out, std::locale(), "{"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to(out, std::locale(), "}"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to(out, std::locale(), "{}"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to(out, std::locale(), "{0}"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to(out, std::locale(), "{:-}", "Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to(out, std::locale(), "{:#}", "Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to(out, std::locale(), "{:L}", "Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to(out, std::locale(), "{0:{0}}", "Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to(out, std::locale(), "{:.42d}", "Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to(out, std::locale(), "{:d}", "Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  std::format_to(wout, std::locale(), L"{"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to(wout, std::locale(), L"}"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to(wout, std::locale(), L"{}"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to(wout, std::locale(), L"{0}"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to(wout, std::locale(), L"{:-}", L"Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to(wout, std::locale(), L"{:#}", L"Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to(wout, std::locale(), L"{:L}", L"Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to(wout, std::locale(), L"{0:{0}}", L"Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to(wout, std::locale(), L"{:.42d}", L"Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}

  std::format_to(wout, std::locale(), L"{:d}", L"Forty-two"); // expected-error-re{{call to consteval function '{{.*}}' is not a constant expression}}
  // expected-note@*:* {{non-constexpr function '__throw_format_error' cannot be used in a constant expression}}
#endif
}
