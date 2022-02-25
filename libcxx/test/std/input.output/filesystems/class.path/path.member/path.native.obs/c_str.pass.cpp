
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

// const value_type* c_str() const noexcept;

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "filesystem_test_helper.h"


int main(int, char**)
{
  using namespace fs;
  const char* const value = "hello world";
  const std::string str_value = value;
  const fs::path::string_type pathstr_value(str_value.begin(), str_value.end());
  { // Check signature
    path p(value);
    ASSERT_SAME_TYPE(path::value_type const*, decltype(p.c_str()));
    ASSERT_NOEXCEPT(p.c_str());
  }
  {
    path p(value);
    assert(p.c_str() == pathstr_value);
    assert(p.native().c_str() == p.c_str());
  }

  return 0;
}
