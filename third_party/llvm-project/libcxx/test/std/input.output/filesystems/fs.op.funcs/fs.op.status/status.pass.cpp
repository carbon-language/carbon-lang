//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// XFAIL: LIBCXX-AIX-FIXME

// <filesystem>

// file_status status(const path& p);
// file_status status(const path& p, error_code& ec) noexcept;

#include "filesystem_include.h"

#include "test_macros.h"
#include "rapid-cxx-test.h"
#include "filesystem_test_helper.h"

using namespace fs;

TEST_SUITE(filesystem_status_test_suite)

TEST_CASE(signature_test)
{
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_NOT_NOEXCEPT(status(p));
    ASSERT_NOEXCEPT(status(p, ec));
}

TEST_CASE(test_status_not_found)
{
    static_test_env static_env;
    const std::errc expect_errc = std::errc::no_such_file_or_directory;
    const path cases[] {
        static_env.DNE,
        static_env.BadSymlink
    };
    for (auto& p : cases) {
        std::error_code ec = std::make_error_code(std::errc::address_in_use);
        // test non-throwing overload.
        file_status st = status(p, ec);
        TEST_CHECK(ErrorIs(ec, expect_errc));
        TEST_CHECK(st.type() == file_type::not_found);
        TEST_CHECK(st.permissions() == perms::unknown);
        // test throwing overload. It should not throw even though it reports
        // that the file was not found.
        TEST_CHECK_NO_THROW(st = status(p));
        TEST_CHECK(st.type() == file_type::not_found);
        TEST_CHECK(st.permissions() == perms::unknown);
    }
}

// Windows doesn't support setting perms::none to trigger failures
// reading directories. Imaginary files under GetWindowsInaccessibleDir()
// produce no_such_file_or_directory, not the error codes this test checks
// for. Finally, status() for a too long file name doesn't return errors
// on windows.
#ifndef TEST_WIN_NO_FILESYSTEM_PERMS_NONE
TEST_CASE(test_status_cannot_resolve)
{
    scoped_test_env env;
    const path dir = env.create_dir("dir");
    const path file = env.create_file("dir/file", 42);
    const path sym = env.create_symlink("dir/file", "sym");
    permissions(dir, perms::none);

    const std::errc set_errc = std::errc::address_in_use;
    const std::errc perm_errc = std::errc::permission_denied;
    const std::errc name_too_long_errc = std::errc::filename_too_long;

    struct TestCase {
      path p;
      std::errc expect_errc;
    } const TestCases[] = {
      {file, perm_errc},
      {sym, perm_errc},
      {path(std::string(2500, 'a')), name_too_long_errc}
    };
    for (auto& TC : TestCases)
    {
        { // test non-throwing case
            std::error_code ec = std::make_error_code(set_errc);
            file_status st = status(TC.p, ec);
            TEST_CHECK(ErrorIs(ec, TC.expect_errc));
            TEST_CHECK(st.type() == file_type::none);
            TEST_CHECK(st.permissions() == perms::unknown);
        }
#ifndef TEST_HAS_NO_EXCEPTIONS
        { // test throwing case
            try {
                status(TC.p);
            } catch (filesystem_error const& err) {
                TEST_CHECK(err.path1() == TC.p);
                TEST_CHECK(err.path2() == "");
                TEST_CHECK(ErrorIs(err.code(), TC.expect_errc));
            }
        }
#endif
    }
}
#endif

TEST_CASE(status_file_types_test)
{
    static_test_env static_env;
    scoped_test_env env;
    struct TestCase {
      path p;
      file_type expect_type;
    } cases[] = {
        {static_env.File, file_type::regular},
        {static_env.SymlinkToFile, file_type::regular},
        {static_env.Dir, file_type::directory},
        {static_env.SymlinkToDir, file_type::directory},
        // file_type::block files tested elsewhere
#ifndef _WIN32
        {static_env.CharFile, file_type::character},
#endif
#if !defined(__APPLE__) && !defined(__FreeBSD__) && !defined(_WIN32) // No support for domain sockets
        {env.create_socket("socket"), file_type::socket},
#endif
#ifndef _WIN32
        {env.create_fifo("fifo"), file_type::fifo}
#endif
    };
    for (const auto& TC : cases) {
        // test non-throwing case
        std::error_code ec = std::make_error_code(std::errc::address_in_use);
        file_status st = status(TC.p, ec);
        TEST_CHECK(!ec);
        TEST_CHECK(st.type() == TC.expect_type);
        TEST_CHECK(st.permissions() != perms::unknown);
        // test throwing case
        TEST_REQUIRE_NO_THROW(st = status(TC.p));
        TEST_CHECK(st.type() == TC.expect_type);
        TEST_CHECK(st.permissions() != perms::unknown);
    }
}

TEST_CASE(test_block_file)
{
    const path possible_paths[] = {
        "/dev/drive0", // Apple
        "/dev/sda",
        "/dev/loop0"
    };
    path p;
    for (const path& possible_p : possible_paths) {
        std::error_code ec;
        if (exists(possible_p, ec)) {
            p = possible_p;
            break;
        }
    }
    if (p == path{}) {
        TEST_UNSUPPORTED();
    }
    // test non-throwing case
    std::error_code ec = std::make_error_code(std::errc::address_in_use);
    file_status st = status(p, ec);
    TEST_CHECK(!ec);
    TEST_CHECK(st.type() == file_type::block);
    TEST_CHECK(st.permissions() != perms::unknown);
    // test throwing case
    TEST_REQUIRE_NO_THROW(st = status(p));
    TEST_CHECK(st.type() == file_type::block);
    TEST_CHECK(st.permissions() != perms::unknown);
}

TEST_SUITE_END()
