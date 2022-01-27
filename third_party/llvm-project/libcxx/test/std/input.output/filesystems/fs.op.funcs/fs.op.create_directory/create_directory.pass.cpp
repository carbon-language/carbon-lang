//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// This test requires the dylib support introduced in D92769.
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.15

// <filesystem>

// bool create_directory(const path& p);
// bool create_directory(const path& p, error_code& ec) noexcept;
// bool create_directory(const path& p, const path& attr);
// bool create_directory(const path& p, const path& attr, error_code& ec) noexcept;

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.h"
#include "filesystem_test_helper.h"

#include <sys/types.h>
#include <sys/stat.h>

using namespace fs;

fs::perms read_umask() {
    auto old_mask = umask(0); // int on Windows, mode_t on POSIX.
    umask(old_mask); // reset the mask to the old value.
    return static_cast<fs::perms>(old_mask);
}

TEST_SUITE(filesystem_create_directory_test_suite)

TEST_CASE(test_signatures)
{
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_SAME_TYPE(decltype(fs::create_directory(p)), bool);
    ASSERT_SAME_TYPE(decltype(fs::create_directory(p, ec)), bool);
    ASSERT_SAME_TYPE(decltype(fs::create_directory(p, p)), bool);
    ASSERT_SAME_TYPE(decltype(fs::create_directory(p, p, ec)), bool);
    ASSERT_NOT_NOEXCEPT(fs::create_directory(p));
    ASSERT_NOEXCEPT(fs::create_directory(p, ec));
    ASSERT_NOT_NOEXCEPT(fs::create_directory(p, p));
    ASSERT_NOEXCEPT(fs::create_directory(p, p, ec));
}


TEST_CASE(create_existing_directory)
{
    scoped_test_env env;
    const path dir = env.create_dir("dir1");
    std::error_code ec;
    TEST_CHECK(fs::create_directory(dir, ec) == false);
    TEST_CHECK(!ec);
    TEST_CHECK(is_directory(dir));
    // Test throwing version
    TEST_CHECK(fs::create_directory(dir) == false);
}

TEST_CASE(create_directory_one_level)
{
    scoped_test_env env;
    const path dir = env.make_env_path("dir1");
    std::error_code ec;
    TEST_CHECK(fs::create_directory(dir, ec) == true);
    TEST_CHECK(!ec);
    TEST_CHECK(is_directory(dir));

    auto st = status(dir);
    const perms expect_perms = perms::all & ~(read_umask());
    TEST_CHECK((st.permissions() & perms::all) == expect_perms);
}

TEST_CASE(create_directory_multi_level)
{
    scoped_test_env env;
    const path dir = env.make_env_path("dir1/dir2");
    const path dir1 = env.make_env_path("dir1");
    std::error_code ec;
    TEST_CHECK(fs::create_directory(dir, ec) == false);
    TEST_CHECK(ec);
    TEST_CHECK(!is_directory(dir));
    TEST_CHECK(!is_directory(dir1));
}

TEST_CASE(dest_is_file)
{
    scoped_test_env env;
    const path file = env.create_file("file", 42);
    std::error_code ec = GetTestEC();
    TEST_CHECK(fs::create_directory(file, ec) == false);
    TEST_CHECK(ec);
    TEST_CHECK(is_regular_file(file));
}

TEST_CASE(dest_part_is_file)
{
    scoped_test_env env;
    const path file = env.create_file("file");
    const path dir = env.make_env_path("file/dir1");
    std::error_code ec = GetTestEC();
    TEST_CHECK(fs::create_directory(dir, ec) == false);
    TEST_CHECK(ec);
    TEST_CHECK(is_regular_file(file));
    TEST_CHECK(!exists(dir));
}

TEST_CASE(dest_is_symlink_to_dir)
{
    scoped_test_env env;
    const path dir = env.create_dir("dir");
    const path sym = env.create_directory_symlink(dir, "sym_name");
    std::error_code ec = GetTestEC();
    TEST_CHECK(create_directory(sym, ec) == false);
    TEST_CHECK(!ec);
}

TEST_CASE(dest_is_symlink_to_file)
{
    scoped_test_env env;
    const path file = env.create_file("file");
    const path sym = env.create_symlink(file, "sym_name");
    std::error_code ec = GetTestEC();
    TEST_CHECK(create_directory(sym, ec) == false);
    TEST_CHECK(ec);
}

TEST_SUITE_END()
