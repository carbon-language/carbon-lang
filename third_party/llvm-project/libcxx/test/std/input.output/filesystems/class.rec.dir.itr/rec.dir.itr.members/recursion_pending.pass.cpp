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

// bool recursion_pending() const;

#include "filesystem_include.h"
#include <type_traits>
#include <set>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.h"
#include "filesystem_test_helper.h"

using namespace fs;

TEST_SUITE(recursive_directory_iterator_recursion_pending_tests)

TEST_CASE(initial_value_test)
{
    static_test_env static_env;
    recursive_directory_iterator it(static_env.Dir);
    TEST_REQUIRE(it.recursion_pending() == true);
}

TEST_CASE(value_after_copy_construction_and_assignment_test)
{
    static_test_env static_env;
    recursive_directory_iterator rec_pending_it(static_env.Dir);
    recursive_directory_iterator no_rec_pending_it(static_env.Dir);
    no_rec_pending_it.disable_recursion_pending();

    { // copy construction
        recursive_directory_iterator it(rec_pending_it);
        TEST_CHECK(it.recursion_pending() == true);
        it.disable_recursion_pending();
        TEST_REQUIRE(rec_pending_it.recursion_pending() == true);

        recursive_directory_iterator it2(no_rec_pending_it);
        TEST_CHECK(it2.recursion_pending() == false);
    }
    { // copy assignment
        recursive_directory_iterator it(static_env.Dir);
        it.disable_recursion_pending();
        it = rec_pending_it;
        TEST_CHECK(it.recursion_pending() == true);
        it.disable_recursion_pending();
        TEST_REQUIRE(rec_pending_it.recursion_pending() == true);

        recursive_directory_iterator it2(static_env.Dir);
        it2 = no_rec_pending_it;
        TEST_CHECK(it2.recursion_pending() == false);
    }
    TEST_CHECK(rec_pending_it.recursion_pending() == true);
    TEST_CHECK(no_rec_pending_it.recursion_pending() == false);
}


TEST_CASE(value_after_move_construction_and_assignment_test)
{
    static_test_env static_env;
    recursive_directory_iterator rec_pending_it(static_env.Dir);
    recursive_directory_iterator no_rec_pending_it(static_env.Dir);
    no_rec_pending_it.disable_recursion_pending();

    { // move construction
        recursive_directory_iterator it_cp(rec_pending_it);
        recursive_directory_iterator it(std::move(it_cp));
        TEST_CHECK(it.recursion_pending() == true);

        recursive_directory_iterator it_cp2(no_rec_pending_it);
        recursive_directory_iterator it2(std::move(it_cp2));
        TEST_CHECK(it2.recursion_pending() == false);
    }
    { // copy assignment
        recursive_directory_iterator it(static_env.Dir);
        it.disable_recursion_pending();
        recursive_directory_iterator it_cp(rec_pending_it);
        it = std::move(it_cp);
        TEST_CHECK(it.recursion_pending() == true);

        recursive_directory_iterator it2(static_env.Dir);
        recursive_directory_iterator it_cp2(no_rec_pending_it);
        it2 = std::move(it_cp2);
        TEST_CHECK(it2.recursion_pending() == false);
    }
    TEST_CHECK(rec_pending_it.recursion_pending() == true);
    TEST_CHECK(no_rec_pending_it.recursion_pending() == false);
}

TEST_CASE(increment_resets_value)
{
    static_test_env static_env;
    const recursive_directory_iterator endIt;
    {
        recursive_directory_iterator it(static_env.Dir);
        it.disable_recursion_pending();
        TEST_CHECK(it.recursion_pending() == false);
        ++it;
        TEST_CHECK(it.recursion_pending() == true);
        TEST_CHECK(it.depth() == 0);
    }
    {
        recursive_directory_iterator it(static_env.Dir);
        it.disable_recursion_pending();
        TEST_CHECK(it.recursion_pending() == false);
        it++;
        TEST_CHECK(it.recursion_pending() == true);
        TEST_CHECK(it.depth() == 0);
    }
    {
        recursive_directory_iterator it(static_env.Dir);
        it.disable_recursion_pending();
        TEST_CHECK(it.recursion_pending() == false);
        std::error_code ec;
        it.increment(ec);
        TEST_CHECK(it.recursion_pending() == true);
        TEST_CHECK(it.depth() == 0);
    }
}

TEST_CASE(pop_does_not_reset_value)
{
    static_test_env static_env;
    const recursive_directory_iterator endIt;

    auto& DE0 = static_env.DirIterationList;
    std::set<path> notSeenDepth0(DE0.begin(), DE0.end());

    recursive_directory_iterator it(static_env.Dir);
    TEST_REQUIRE(it != endIt);

    while (it.depth() == 0) {
        notSeenDepth0.erase(it->path());
        ++it;
        TEST_REQUIRE(it != endIt);
    }
    TEST_REQUIRE(it.depth() == 1);
    it.disable_recursion_pending();
    it.pop();
    // Since the order of iteration is unspecified the pop() could result
    // in the end iterator. When this is the case it is undefined behavior
    // to call recursion_pending().
    if (it == endIt) {
        TEST_CHECK(notSeenDepth0.empty());
#if defined(_LIBCPP_VERSION)
        TEST_CHECK(it.recursion_pending() == false);
#endif
    } else {
        TEST_CHECK(! notSeenDepth0.empty());
        TEST_CHECK(it.recursion_pending() == false);
    }
}

TEST_SUITE_END()
