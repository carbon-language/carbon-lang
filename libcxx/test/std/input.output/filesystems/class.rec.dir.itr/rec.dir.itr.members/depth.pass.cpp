//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// class recursive_directory_iterator

// int depth() const

#include "filesystem_include.h"
#include <type_traits>
#include <set>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.h"
#include "filesystem_test_helper.h"

using namespace fs;

TEST_SUITE(recursive_directory_iterator_depth_tests)

TEST_CASE(test_depth)
{
    static_test_env static_env;
    const path testDir = static_env.Dir;
    const path DirDepth1 = static_env.Dir2;
    const path DirDepth2 = static_env.Dir3;
    const recursive_directory_iterator endIt{};

    std::error_code ec;
    recursive_directory_iterator it(testDir, ec);
    TEST_REQUIRE(!ec);
    TEST_CHECK(it.depth() == 0);

    bool seen_d1, seen_d2;
    seen_d1 = seen_d2 = false;

    while (it != endIt) {
        const path entry = *it;
        const path parent = entry.parent_path();
        if (parent == testDir) {
            TEST_CHECK(it.depth() == 0);
        } else if (parent == DirDepth1) {
            TEST_CHECK(it.depth() == 1);
            seen_d1 = true;
        } else if (parent == DirDepth2) {
            TEST_CHECK(it.depth() == 2);
            seen_d2 = true;
        } else {
            TEST_CHECK(!"Unexpected depth while iterating over static env");
        }
        ++it;
    }
    TEST_REQUIRE(seen_d1 && seen_d2);
    TEST_CHECK(it == endIt);
}

TEST_SUITE_END()
