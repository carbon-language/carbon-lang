//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// path absolute(const path& p, const path& base=current_path());

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.h"
#include "filesystem_test_helper.h"

using namespace fs;

TEST_SUITE(filesystem_absolute_path_test_suite)

TEST_CASE(absolute_signature_test)
{
    const path p; ((void)p);
    std::error_code ec;
    ASSERT_NOT_NOEXCEPT(absolute(p));
    ASSERT_NOT_NOEXCEPT(absolute(p, ec));
}


TEST_CASE(basic_test)
{
    const fs::path cwd = fs::current_path();
    const struct {
      std::string input;
      fs::path expect;
    } TestCases [] = {
        {"", cwd / ""},
        {"foo", cwd / "foo"},
        {"foo/", cwd / "foo/"},
        {"/already_absolute", cwd.root_name() / "/already_absolute"}
    };
    for (auto& TC : TestCases) {
        std::error_code ec = GetTestEC();
        const path ret = absolute(TC.input, ec);
        TEST_CHECK(!ec);
        TEST_CHECK(ret.is_absolute());
        TEST_CHECK(PathEqIgnoreSep(ret, TC.expect));
        LIBCPP_ONLY(TEST_CHECK(PathEq(ret, TC.expect)));
    }
}

TEST_SUITE_END()
