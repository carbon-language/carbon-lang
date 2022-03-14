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

// directory_iterator(directory_iterator const&);

#include "filesystem_include.h"
#include <type_traits>
#include <set>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.h"
#include "filesystem_test_helper.h"

using namespace fs;

TEST_SUITE(directory_iterator_copy_construct_tests)

TEST_CASE(test_constructor_signature)
{
    using D = directory_iterator;
    static_assert(std::is_copy_constructible<D>::value, "");
}

TEST_CASE(test_copy_end_iterator)
{
    const directory_iterator endIt;
    directory_iterator it(endIt);
    TEST_CHECK(it == endIt);
}

TEST_CASE(test_copy_valid_iterator)
{
    static_test_env static_env;
    const path testDir = static_env.Dir;
    const directory_iterator endIt{};

    const directory_iterator it(testDir);
    TEST_REQUIRE(it != endIt);
    const path entry = *it;

    const directory_iterator it2(it);
    TEST_REQUIRE(it2 == it);
    TEST_CHECK(*it2 == entry);
    TEST_CHECK(*it == entry);
}

TEST_SUITE_END()
