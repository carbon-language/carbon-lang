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

// template <class T> constexpr bool is_error_condition_enum_v;

#include <experimental/system_error>
#include <type_traits>
namespace ex = std::experimental;

int main() {
  {
    static_assert(ex::is_error_condition_enum_v<std::errc>, "");

    static_assert(ex::is_error_condition_enum_v<std::errc> ==
                  std::is_error_condition_enum <std::errc>::value, "");

    static_assert(
        std::is_same<decltype(ex::is_error_condition_enum_v<std::errc>),
                     const bool>::value,
        "");
  }
  {
    static_assert(!ex::is_error_condition_enum_v<int>, "");

    static_assert(ex::is_error_condition_enum_v<int> ==
                  std::is_error_condition_enum <int>::value, "");
  }
}
