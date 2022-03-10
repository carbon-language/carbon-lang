//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// class directory_iterator

// directory_iterator& operator=(directory_iterator const&);

#include "filesystem_include.h"
#include <type_traits>
#include <set>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.h"
#include "filesystem_test_helper.h"

// The filesystem specification explicitly allows for self-move on
// the directory iterators. Turn off this warning so we can test it.
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wself-move"
#endif

using namespace fs;

TEST_SUITE(directory_iterator_move_assign_tests)

TEST_CASE(test_assignment_signature)
{
    using D = directory_iterator;
    static_assert(std::is_nothrow_move_assignable<D>::value, "");
}

TEST_CASE(test_move_to_end_iterator)
{
    static_test_env static_env;
    const path testDir = static_env.Dir;

    directory_iterator from(testDir);
    TEST_REQUIRE(from != directory_iterator{});
    const path entry = *from;

    directory_iterator to{};
    to = std::move(from);
    TEST_REQUIRE(to != directory_iterator{});
    TEST_CHECK(*to == entry);
}


TEST_CASE(test_move_from_end_iterator)
{
    static_test_env static_env;
    const path testDir = static_env.Dir;

    directory_iterator from{};

    directory_iterator to(testDir);
    TEST_REQUIRE(to != from);

    to = std::move(from);
    TEST_REQUIRE(to == directory_iterator{});
    TEST_REQUIRE(from == directory_iterator{});
}

TEST_CASE(test_move_valid_iterator)
{
    static_test_env static_env;
    const path testDir = static_env.Dir;
    const directory_iterator endIt{};

    directory_iterator it(testDir);
    TEST_REQUIRE(it != endIt);
    ++it;
    TEST_REQUIRE(it != endIt);
    const path entry = *it;

    directory_iterator it2(testDir);
    TEST_REQUIRE(it2 != it);
    const path entry2 = *it2;
    TEST_CHECK(entry2 != entry);

    it2 = std::move(it);
    TEST_REQUIRE(it2 != directory_iterator{});
    TEST_CHECK(*it2 == entry);
}

TEST_CASE(test_returns_reference_to_self)
{
    directory_iterator it;
    directory_iterator it2;
    directory_iterator& ref = (it2 = it);
    TEST_CHECK(&ref == &it2);
}


TEST_CASE(test_self_move)
{
    static_test_env static_env;
    // Create two non-equal iterators that have exactly the same state.
    directory_iterator it(static_env.Dir);
    directory_iterator it2(static_env.Dir);
    ++it; ++it2;
    TEST_CHECK(it != it2);
    TEST_CHECK(*it2 == *it);

    it = std::move(it);
    TEST_CHECK(*it2 == *it);
}


TEST_SUITE_END()
