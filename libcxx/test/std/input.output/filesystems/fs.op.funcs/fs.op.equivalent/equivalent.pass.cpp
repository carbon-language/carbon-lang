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

// bool equivalent(path const& lhs, path const& rhs);
// bool equivalent(path const& lhs, path const& rhs, std::error_code& ec) noexcept;

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.h"
#include "filesystem_test_helper.h"

using namespace fs;

TEST_SUITE(equivalent_test_suite)

TEST_CASE(signature_test) {
  const path p;
  ((void)p);
  std::error_code ec;
  ((void)ec);
  ASSERT_NOEXCEPT(equivalent(p, p, ec));
  ASSERT_NOT_NOEXCEPT(equivalent(p, p));
}

TEST_CASE(equivalent_test) {
  static_test_env static_env;
  struct TestCase {
    path lhs;
    path rhs;
    bool expect;
  };
  const TestCase testCases[] = {
      {static_env.Dir, static_env.Dir, true},
      {static_env.File, static_env.Dir, false},
      {static_env.Dir, static_env.SymlinkToDir, true},
      {static_env.Dir, static_env.SymlinkToFile, false},
      {static_env.File, static_env.File, true},
      {static_env.File, static_env.SymlinkToFile, true},
  };
  for (auto& TC : testCases) {
    std::error_code ec;
    TEST_CHECK(equivalent(TC.lhs, TC.rhs, ec) == TC.expect);
    TEST_CHECK(!ec);
  }
}

TEST_CASE(equivalent_reports_error_if_input_dne) {
  static_test_env static_env;
  const path E = static_env.File;
  const path DNE = static_env.DNE;
  { // Test that an error is reported when either of the paths don't exist
    std::error_code ec = GetTestEC();
    TEST_CHECK(equivalent(E, DNE, ec) == false);
    TEST_CHECK(ec);
    TEST_CHECK(ec != GetTestEC());
  }
  {
    std::error_code ec = GetTestEC();
    TEST_CHECK(equivalent(DNE, E, ec) == false);
    TEST_CHECK(ec);
    TEST_CHECK(ec != GetTestEC());
  }
  {
    TEST_CHECK_THROW(filesystem_error, equivalent(DNE, E));
    TEST_CHECK_THROW(filesystem_error, equivalent(E, DNE));
  }
  { // Test that an exception is thrown if both paths do not exist.
    TEST_CHECK_THROW(filesystem_error, equivalent(DNE, DNE));
  }
  {
    std::error_code ec = GetTestEC();
    TEST_CHECK(equivalent(DNE, DNE, ec) == false);
    TEST_CHECK(ec);
    TEST_CHECK(ec != GetTestEC());
  }
}

TEST_CASE(equivalent_hardlink_succeeds) {
  scoped_test_env env;
  path const file = env.create_file("file", 42);
  const path hl1 = env.create_hardlink(file, "hl1");
  const path hl2 = env.create_hardlink(file, "hl2");
  TEST_CHECK(equivalent(file, hl1));
  TEST_CHECK(equivalent(file, hl2));
  TEST_CHECK(equivalent(hl1, hl2));
}

#ifndef _WIN32
TEST_CASE(equivalent_is_other_succeeds) {
  scoped_test_env env;
  path const file = env.create_file("file", 42);
  const path fifo1 = env.create_fifo("fifo1");
  const path fifo2 = env.create_fifo("fifo2");
  // Required to test behavior for inputs where is_other(p) is true.
  TEST_REQUIRE(is_other(fifo1));
  TEST_CHECK(!equivalent(file, fifo1));
  TEST_CHECK(!equivalent(fifo2, file));
  TEST_CHECK(!equivalent(fifo1, fifo2));
  TEST_CHECK(equivalent(fifo1, fifo1));
}
#endif

TEST_SUITE_END()
