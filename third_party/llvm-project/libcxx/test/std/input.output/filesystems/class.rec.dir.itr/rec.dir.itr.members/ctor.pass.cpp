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

// class directory_iterator

//
// explicit recursive_directory_iterator(const path& p);
// recursive_directory_iterator(const path& p, directory_options options);
// recursive_directory_iterator(const path& p, error_code& ec);
// recursive_directory_iterator(const path& p, directory_options options, error_code& ec);


#include "filesystem_include.h"
#include <type_traits>
#include <set>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.h"
#include "filesystem_test_helper.h"

using namespace fs;

using RDI = recursive_directory_iterator;

TEST_SUITE(recursive_directory_iterator_constructor_tests)

TEST_CASE(test_constructor_signatures)
{
    using D = recursive_directory_iterator;

    // explicit directory_iterator(path const&);
    static_assert(!std::is_convertible<path, D>::value, "");
    static_assert(std::is_constructible<D, path>::value, "");
    static_assert(!std::is_nothrow_constructible<D, path>::value, "");

    // directory_iterator(path const&, error_code&)
    static_assert(std::is_constructible<D, path,
        std::error_code&>::value, "");
    static_assert(!std::is_nothrow_constructible<D, path,
        std::error_code&>::value, "");

    // directory_iterator(path const&, directory_options);
    static_assert(std::is_constructible<D, path, directory_options>::value, "");
    static_assert(!std::is_nothrow_constructible<D, path, directory_options>::value, "");

    // directory_iterator(path const&, directory_options, error_code&)
    static_assert(std::is_constructible<D, path, directory_options, std::error_code&>::value, "");
    static_assert(!std::is_nothrow_constructible<D, path, directory_options, std::error_code&>::value, "");
}

TEST_CASE(test_construction_from_bad_path)
{
    static_test_env static_env;
    std::error_code ec;
    directory_options opts = directory_options::none;
    const RDI endIt;

    const path testPaths[] = { static_env.DNE, static_env.BadSymlink };
    for (path const& testPath : testPaths)
    {
        {
            RDI it(testPath, ec);
            TEST_CHECK(ec);
            TEST_CHECK(it == endIt);
        }
        {
            RDI it(testPath, opts, ec);
            TEST_CHECK(ec);
            TEST_CHECK(it == endIt);
        }
        {
            TEST_CHECK_THROW(filesystem_error, RDI(testPath));
            TEST_CHECK_THROW(filesystem_error, RDI(testPath, opts));
        }
    }
}

TEST_CASE(access_denied_test_case)
{
    using namespace fs;
#ifdef _WIN32
    // Windows doesn't support setting perms::none to trigger failures
    // reading directories; test using a special inaccessible directory
    // instead.
    const path testDir = GetWindowsInaccessibleDir();
    if (testDir.empty())
        TEST_UNSUPPORTED();
#else
    scoped_test_env env;
    path const testDir = env.make_env_path("dir1");
    path const testFile = testDir / "testFile";
    env.create_dir(testDir);
    env.create_file(testFile, 42);

    // Test that we can iterator over the directory before changing the perms
    {
        RDI it(testDir);
        TEST_REQUIRE(it != RDI{});
    }

    // Change the permissions so we can no longer iterate
    permissions(testDir, perms::none);
#endif

    // Check that the construction fails when skip_permissions_denied is
    // not given.
    {
        std::error_code ec;
        RDI it(testDir, ec);
        TEST_REQUIRE(ec);
        TEST_CHECK(it == RDI{});
    }
    // Check that construction does not report an error when
    // 'skip_permissions_denied' is given.
    {
        std::error_code ec;
        RDI it(testDir, directory_options::skip_permission_denied, ec);
        TEST_REQUIRE(!ec);
        TEST_CHECK(it == RDI{});
    }
}


TEST_CASE(access_denied_to_file_test_case)
{
    using namespace fs;
#ifdef _WIN32
    // Windows doesn't support setting perms::none to trigger failures
    // reading directories; test using a special inaccessible directory
    // instead.
    const path testDir = GetWindowsInaccessibleDir();
    if (testDir.empty())
        TEST_UNSUPPORTED();
    path const testFile = testDir / "inaccessible_file";
#else
    scoped_test_env env;
    path const testFile = env.make_env_path("file1");
    env.create_file(testFile, 42);

    // Change the permissions so we can no longer iterate
    permissions(testFile, perms::none);
#endif

    // Check that the construction fails when skip_permissions_denied is
    // not given.
    {
        std::error_code ec;
        RDI it(testFile, ec);
        TEST_REQUIRE(ec);
        TEST_CHECK(it == RDI{});
    }
    // Check that construction still fails when 'skip_permissions_denied' is given
    // because we tried to open a file and not a directory.
    {
        std::error_code ec;
        RDI it(testFile, directory_options::skip_permission_denied, ec);
        TEST_REQUIRE(ec);
        TEST_CHECK(it == RDI{});
    }
}

TEST_CASE(test_open_on_empty_directory_equals_end)
{
    scoped_test_env env;
    const path testDir = env.make_env_path("dir1");
    env.create_dir(testDir);

    const RDI endIt;
    {
        std::error_code ec;
        RDI it(testDir, ec);
        TEST_CHECK(!ec);
        TEST_CHECK(it == endIt);
    }
    {
        RDI it(testDir);
        TEST_CHECK(it == endIt);
    }
}

TEST_CASE(test_open_on_directory_succeeds)
{
    static_test_env static_env;
    const path testDir = static_env.Dir;
    std::set<path> dir_contents(static_env.DirIterationList.begin(),
                                static_env.DirIterationList.end());
    const RDI endIt{};

    {
        std::error_code ec;
        RDI it(testDir, ec);
        TEST_REQUIRE(!ec);
        TEST_CHECK(it != endIt);
        TEST_CHECK(dir_contents.count(*it));
    }
    {
        RDI it(testDir);
        TEST_CHECK(it != endIt);
        TEST_CHECK(dir_contents.count(*it));
    }
}

TEST_CASE(test_open_on_file_fails)
{
    static_test_env static_env;
    const path testFile = static_env.File;
    const RDI endIt{};
    {
        std::error_code ec;
        RDI it(testFile, ec);
        TEST_REQUIRE(ec);
        TEST_CHECK(it == endIt);
    }
    {
        TEST_CHECK_THROW(filesystem_error, RDI(testFile));
    }
}

TEST_CASE(test_options_post_conditions)
{
    static_test_env static_env;
    const path goodDir = static_env.Dir;
    const path badDir = static_env.DNE;

    {
        std::error_code ec;

        RDI it1(goodDir, ec);
        TEST_REQUIRE(!ec);
        TEST_CHECK(it1.options() == directory_options::none);

        RDI it2(badDir, ec);
        TEST_REQUIRE(ec);
        TEST_REQUIRE(it2 == RDI{});
    }
    {
        std::error_code ec;
        const directory_options opts = directory_options::skip_permission_denied;

        RDI it1(goodDir, opts, ec);
        TEST_REQUIRE(!ec);
        TEST_CHECK(it1.options() == opts);

        RDI it2(badDir, opts, ec);
        TEST_REQUIRE(ec);
        TEST_REQUIRE(it2 == RDI{});
    }
    {
        RDI it(goodDir);
        TEST_CHECK(it.options() == directory_options::none);
    }
    {
        const directory_options opts = directory_options::follow_directory_symlink;
        RDI it(goodDir, opts);
        TEST_CHECK(it.options() == opts);
    }
}
TEST_SUITE_END()
