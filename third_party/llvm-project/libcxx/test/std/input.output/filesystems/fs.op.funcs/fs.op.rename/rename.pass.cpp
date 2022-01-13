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

// void rename(const path& old_p, const path& new_p);
// void rename(const path& old_p,  const path& new_p, error_code& ec) noexcept;

#include "filesystem_include.h"

#include "test_macros.h"
#include "rapid-cxx-test.h"
#include "filesystem_test_helper.h"

using namespace fs;

TEST_SUITE(filesystem_rename_test_suite)

TEST_CASE(test_signatures)
{
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_SAME_TYPE(decltype(fs::rename(p, p)), void);
    ASSERT_SAME_TYPE(decltype(fs::rename(p, p, ec)), void);

    ASSERT_NOT_NOEXCEPT(fs::rename(p, p));
    ASSERT_NOEXCEPT(fs::rename(p, p, ec));
}

TEST_CASE(test_error_reporting)
{
    auto checkThrow = [](path const& f, path const& t, const std::error_code& ec)
    {
#ifndef TEST_HAS_NO_EXCEPTIONS
        try {
            fs::rename(f, t);
            return false;
        } catch (filesystem_error const& err) {
            return err.path1() == f
                && err.path2() == t
                && err.code() == ec;
        }
#else
        ((void)f); ((void)t); ((void)ec);
        return true;
#endif
    };
    scoped_test_env env;
    const path dne = env.make_env_path("dne");
    const path file = env.create_file("file1", 42);
    const path dir = env.create_dir("dir1");
    struct TestCase {
      path from;
      path to;
    } cases[] = {
        {dne, dne},
        {file, dir},
#ifndef _WIN32
        // The spec doesn't say that this case must be an error; fs.op.rename
        // note 1.2.1 says that a file may be overwritten by a rename.
        // On Windows, with rename() implemented with MoveFileExW, overwriting
        // a file with a directory is not an error.
        {dir, file},
#endif
    };
    for (auto& TC : cases) {
        auto from_before = status(TC.from);
        auto to_before = status(TC.to);
        std::error_code ec;
        rename(TC.from, TC.to, ec);
        TEST_REQUIRE(ec);
        TEST_CHECK(from_before.type() == status(TC.from).type());
        TEST_CHECK(to_before.type() == status(TC.to).type());
        TEST_CHECK(checkThrow(TC.from, TC.to, ec));
    }
}

TEST_CASE(basic_rename_test)
{
    scoped_test_env env;

    const std::error_code set_ec = std::make_error_code(std::errc::address_in_use);
    const path file = env.create_file("file1", 42);
    { // same file
        std::error_code ec = set_ec;
        rename(file, file, ec);
        TEST_CHECK(!ec);
        TEST_CHECK(is_regular_file(file));
        TEST_CHECK(file_size(file) == 42);
    }
    const path sym = env.create_symlink(file, "sym");
    { // file -> symlink
        std::error_code ec = set_ec;
        rename(file, sym, ec);
        TEST_CHECK(!ec);
        TEST_CHECK(!exists(file));
        TEST_CHECK(is_regular_file(symlink_status(sym)));
        TEST_CHECK(file_size(sym) == 42);
    }
    const path file2 = env.create_file("file2", 42);
    const path file3 = env.create_file("file3", 100);
    { // file -> file
        std::error_code ec = set_ec;
        rename(file2, file3, ec);
        TEST_CHECK(!ec);
        TEST_CHECK(!exists(file2));
        TEST_CHECK(is_regular_file(file3));
        TEST_CHECK(file_size(file3) == 42);
    }
    const path dne = env.make_env_path("dne");
    const path bad_sym = env.create_symlink(dne, "bad_sym");
    const path bad_sym_dest = env.make_env_path("bad_sym2");
    { // bad-symlink
        std::error_code ec = set_ec;
        rename(bad_sym, bad_sym_dest, ec);
        TEST_CHECK(!ec);
        TEST_CHECK(!exists(symlink_status(bad_sym)));
        TEST_CHECK(is_symlink(bad_sym_dest));
        TEST_CHECK(read_symlink(bad_sym_dest) == dne);
    }
}

TEST_CASE(basic_rename_dir_test)
{
    static_test_env env;
    const std::error_code set_ec = std::make_error_code(std::errc::address_in_use);
    const path new_dir = env.makePath("new_dir");
    { // dir -> dir (with contents)
        std::error_code ec = set_ec;
        rename(env.Dir, new_dir, ec);
        TEST_CHECK(!ec);
        TEST_CHECK(!exists(env.Dir));
        TEST_CHECK(is_directory(new_dir));
        TEST_CHECK(exists(new_dir / "file1"));
    }
#ifdef _WIN32
    // On Windows, renaming a directory over a file isn't an error (this
    // case is skipped in test_error_reporting above).
    { // dir -> file
        std::error_code ec = set_ec;
        rename(new_dir, env.NonEmptyFile, ec);
        TEST_CHECK(!ec);
        TEST_CHECK(!exists(new_dir));
        TEST_CHECK(is_directory(env.NonEmptyFile));
        TEST_CHECK(exists(env.NonEmptyFile / "file1"));
    }
#endif
}

TEST_SUITE_END()
