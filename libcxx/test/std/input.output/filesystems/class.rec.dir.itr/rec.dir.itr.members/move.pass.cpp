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

// class recursive_directory_iterator

// recursive_directory_iterator(recursive_directory_iterator&&) noexcept;

#include "filesystem_include.h"
#include <type_traits>
#include <set>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.h"
#include "filesystem_test_helper.h"

using namespace fs;

TEST_SUITE(recursive_directory_iterator_move_construct_tests)

TEST_CASE(test_constructor_signature)
{
    using D = recursive_directory_iterator;
    static_assert(std::is_nothrow_move_constructible<D>::value, "");
}

TEST_CASE(test_move_end_iterator)
{
    const recursive_directory_iterator endIt;
    recursive_directory_iterator endIt2{};

    recursive_directory_iterator it(std::move(endIt2));
    TEST_CHECK(it == endIt);
    TEST_CHECK(endIt2 == endIt);
}

TEST_CASE(test_move_valid_iterator)
{
    static_test_env static_env;
    const path testDir = static_env.Dir;
    const recursive_directory_iterator endIt{};

    // build 'it' up with "interesting" non-default state so we can test
    // that it gets copied. We want to get 'it' into a state such that:
    //  it.options() != directory_options::none
    //  it.depth() != 0
    //  it.recursion_pending() != true
    const directory_options opts = directory_options::skip_permission_denied;
    recursive_directory_iterator it(testDir, opts);
    TEST_REQUIRE(it != endIt);
    while (it.depth() == 0) {
        ++it;
        TEST_REQUIRE(it != endIt);
    }
    it.disable_recursion_pending();
    TEST_CHECK(it.options() == opts);
    TEST_CHECK(it.depth() == 1);
    TEST_CHECK(it.recursion_pending() == false);
    const path entry = *it;

    // OPERATION UNDER TEST //
    const recursive_directory_iterator it2(std::move(it));
    // ------------------- //

    TEST_REQUIRE(it2 != endIt);
    TEST_CHECK(*it2 == entry);
    TEST_CHECK(it2.depth() == 1);
    TEST_CHECK(it2.recursion_pending() == false);
}

TEST_SUITE_END()
