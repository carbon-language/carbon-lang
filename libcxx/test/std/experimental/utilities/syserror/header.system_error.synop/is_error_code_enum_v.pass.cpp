//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <experimental/system_error>

// template <class T> constexpr bool is_error_code_enum_v;

#include <experimental/system_error>
#include <ios> /* for std::io_errc */

namespace ex = std::experimental;

int main() {
  {
    static_assert(ex::is_error_code_enum_v<std::io_errc>, "");

    static_assert(ex::is_error_code_enum_v<std::io_errc> ==
                  std::is_error_code_enum <std::io_errc>::value, "");

    static_assert(std::is_same<decltype(ex::is_error_code_enum_v<std::io_errc>),
                               const bool>::value, "");
  }
  {
    static_assert(!ex::is_error_code_enum_v<int>, "");

    static_assert(ex::is_error_code_enum_v<int> ==
                  std::is_error_code_enum <int>::value, "");
  }
}
