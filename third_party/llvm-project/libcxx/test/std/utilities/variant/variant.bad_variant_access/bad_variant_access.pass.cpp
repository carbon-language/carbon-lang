//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// Throwing bad_variant_access is supported starting in macosx10.13
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12}}

// <variant>

/*

 class bad_variant_access : public exception {
public:
  bad_variant_access() noexcept;
  virtual const char* what() const noexcept;
};

*/

#include <cassert>
#include <exception>
#include <type_traits>
#include <variant>

#include "test_macros.h"

int main(int, char**) {
  static_assert(std::is_base_of<std::exception, std::bad_variant_access>::value,
                "");
  static_assert(noexcept(std::bad_variant_access{}), "must be noexcept");
  static_assert(noexcept(std::bad_variant_access{}.what()), "must be noexcept");
  std::bad_variant_access ex;
  assert(ex.what());

  return 0;
}
