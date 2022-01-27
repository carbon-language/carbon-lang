//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// void create_directory_symlink(const path& existing_symlink, const path& new_symlink);
// void create_directory_symlink(const path& existing_symlink, const path& new_symlink,
//                   error_code& ec) noexcept;

#include "filesystem_include.h"
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.h"
#include "filesystem_test_helper.h"

using namespace fs;

TEST_SUITE(filesystem_create_directory_symlink_test_suite)

TEST_CASE(test_signatures)
{
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_NOT_NOEXCEPT(fs::create_directory_symlink(p, p));
    ASSERT_NOEXCEPT(fs::create_directory_symlink(p, p, ec));
}

TEST_CASE(test_error_reporting)
{
    scoped_test_env env;
    const path file = env.create_file("file1", 42);
    const path file2 = env.create_file("file2", 55);
    const path sym = env.create_symlink(file, "sym");
    { // destination exists
        std::error_code ec;
        fs::create_directory_symlink(sym, file2, ec);
        TEST_REQUIRE(ec);
    }
}

TEST_CASE(create_directory_symlink_basic)
{
    scoped_test_env env;
    const path dir = env.create_dir("dir");
    const path dir_sym = env.create_directory_symlink(dir, "dir_sym");

    const path dest = env.make_env_path("dest1");
    std::error_code ec;
    fs::create_directory_symlink(dir_sym, dest, ec);
    TEST_REQUIRE(!ec);
    TEST_CHECK(is_symlink(dest));
    TEST_CHECK(equivalent(dest, dir));
}


TEST_SUITE_END()
