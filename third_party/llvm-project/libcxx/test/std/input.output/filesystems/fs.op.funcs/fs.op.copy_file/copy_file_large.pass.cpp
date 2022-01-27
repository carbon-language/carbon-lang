//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// REQUIRES: long_tests

// <filesystem>

// bool copy_file(const path& from, const path& to);
// bool copy_file(const path& from, const path& to, error_code& ec) noexcept;
// bool copy_file(const path& from, const path& to, copy_options options);
// bool copy_file(const path& from, const path& to, copy_options options,
//           error_code& ec) noexcept;

#include "filesystem_include.h"
#include <cassert>
#include <cstdio>
#include <string>

#include "test_macros.h"
#include "rapid-cxx-test.h"
#include "filesystem_test_helper.h"

using namespace fs;

TEST_SUITE(filesystem_copy_file_test_suite)

// This test is intended to test 'sendfile's 2gb limit for a single call, and
// to ensure that libc++ correctly copies files larger than that limit.
// However it requires allocating ~5GB of filesystem space. This might not
// be acceptable on all systems.
TEST_CASE(large_file) {
  using namespace fs;
  constexpr uintmax_t sendfile_size_limit = 2147479552ull;
  constexpr uintmax_t additional_size = 1024;
  constexpr uintmax_t test_file_size = sendfile_size_limit + additional_size;
  static_assert(test_file_size > sendfile_size_limit, "");

  scoped_test_env env;

  // Check that we have more than sufficient room to create the files needed
  // to perform the test.
  if (space(env.test_root).available < 3 * test_file_size) {
    TEST_UNSUPPORTED();
  }

  // Create a file right at the size limit. The file is full of '\0's.
  const path source = env.create_file("source", sendfile_size_limit);
  const std::string additional_data(additional_size, 'x');
  // Append known data to the end of the source file.
  {
    std::FILE* outf = std::fopen(source.string().c_str(), "a");
    TEST_REQUIRE(outf != nullptr);
    std::fputs(additional_data.c_str(), outf);
    std::fclose(outf);
  }
  TEST_REQUIRE(file_size(source) == test_file_size);
  const path dest = env.make_env_path("dest");

  std::error_code ec = GetTestEC();
  TEST_CHECK(copy_file(source, dest, ec));
  TEST_CHECK(!ec);

  TEST_REQUIRE(is_regular_file(dest));
  TEST_CHECK(file_size(dest) == test_file_size);

  // Read the data from the end of the destination file, and ensure it matches
  // the data at the end of the source file.
  std::string out_data(additional_size, 'z');
  {
    std::FILE* dest_file = std::fopen(dest.string().c_str(), "rb");
    TEST_REQUIRE(dest_file != nullptr);
    TEST_REQUIRE(std::fseek(dest_file, sendfile_size_limit, SEEK_SET) == 0);
    TEST_REQUIRE(std::fread(&out_data[0], sizeof(out_data[0]), additional_size, dest_file) == additional_size);
    std::fclose(dest_file);
  }
  TEST_CHECK(out_data.size() == additional_data.size());
  TEST_CHECK(out_data == additional_data);
}

TEST_SUITE_END()
