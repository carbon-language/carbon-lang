//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// class path

// const string_type& native() const noexcept;

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "filesystem_test_helper.h"


int main(int, char**)
{
  using namespace fs;
  const char* const value = "hello world";
  std::string value_str(value);
  fs::path::string_type pathstr_value(value_str.begin(), value_str.end());
  { // Check signature
    path p(value);
    ASSERT_SAME_TYPE(path::string_type const&, decltype(p.native()));
    ASSERT_NOEXCEPT(p.native());
  }
  { // native() is tested elsewhere
    path p(value);
    assert(p.native() == pathstr_value);
  }

  return 0;
}
