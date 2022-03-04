//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// space_info space(const path& p);
// space_info space(const path& p, error_code& ec) noexcept;

#include "filesystem_include.h"

#include "test_macros.h"
#include "rapid-cxx-test.h"
#include "filesystem_test_helper.h"

using namespace fs;

bool EqualDelta(std::uintmax_t x, std::uintmax_t y, std::uintmax_t delta) {
    if (x >= y) {
        return (x - y) <= delta;
    } else {
        return (y - x) <= delta;
    }
}

TEST_SUITE(filesystem_space_test_suite)

TEST_CASE(signature_test)
{
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_SAME_TYPE(decltype(space(p)), space_info);
    ASSERT_SAME_TYPE(decltype(space(p, ec)), space_info);
    ASSERT_NOT_NOEXCEPT(space(p));
    ASSERT_NOEXCEPT(space(p, ec));
}

TEST_CASE(test_error_reporting)
{
    static_test_env static_env;
    auto checkThrow = [](path const& f, const std::error_code& ec)
    {
#ifndef TEST_HAS_NO_EXCEPTIONS
        try {
            (void)space(f);
            return false;
        } catch (filesystem_error const& err) {
            return err.path1() == f
                && err.path2() == ""
                && err.code() == ec;
        }
#else
        ((void)f); ((void)ec);
        return true;
#endif
    };
    const path cases[] = {
        "",
        static_env.DNE,
        static_env.BadSymlink
    };
    for (auto& p : cases) {
        const auto expect = static_cast<std::uintmax_t>(-1);
        std::error_code ec;
        space_info info = space(p, ec);
        TEST_CHECK(ec);
        TEST_CHECK(info.capacity == expect);
        TEST_CHECK(info.free == expect);
        TEST_CHECK(info.available == expect);
        TEST_CHECK(checkThrow(p, ec));
    }
}

TEST_CASE(basic_space_test)
{
    static_test_env static_env;

    // All the test cases should reside on the same filesystem and therefore
    // should have the same expected result. Compute this expected result
    // one and check that it looks semi-sane.
    const std::uintmax_t bad_value = static_cast<std::uintmax_t>(-1);
    std::uintmax_t expect_capacity;
    std::uintmax_t expect_free;
    std::uintmax_t expect_avail;
    TEST_REQUIRE(utils::space(static_env.Dir.string(), expect_capacity,
                              expect_free, expect_avail));

    // Other processes running on the operating system may have changed
    // the amount of space available. Check that these are within tolerances.
    // Currently 5% of capacity
    const std::uintmax_t delta = expect_capacity / 20;
    const path cases[] = {
        static_env.File,
        static_env.Dir,
        static_env.Dir2,
        static_env.SymlinkToFile,
        static_env.SymlinkToDir
    };
    for (auto& p : cases) {
        std::error_code ec = GetTestEC();
        space_info info = space(p, ec);
        TEST_CHECK(!ec);
        TEST_CHECK(info.capacity != bad_value);
        TEST_CHECK(expect_capacity == info.capacity);
        TEST_CHECK(info.free != bad_value);
        TEST_CHECK(EqualDelta(expect_free, info.free, delta));
        TEST_CHECK(info.available != bad_value);
        TEST_CHECK(EqualDelta(expect_avail, info.available, delta));
    }
}

TEST_SUITE_END()
