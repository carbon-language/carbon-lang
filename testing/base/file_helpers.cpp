// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/base/file_helpers.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <system_error>

namespace Carbon::Testing {

auto ReadFile(std::filesystem::path path) -> ErrorOr<std::string> {
  std::ifstream file_stream(path);
  if (file_stream.fail()) {
    return Error(llvm::formatv("Error opening file: {0}", path));
  }
  std::stringstream buffer;
  buffer << file_stream.rdbuf();
  if (file_stream.fail()) {
    return Error(llvm::formatv("Error reading file: {0}", path));
  }
  return buffer.str();
}

auto WriteTestFile(llvm::StringRef name, llvm::StringRef contents)
    -> ErrorOr<std::filesystem::path> {
  std::filesystem::path test_tmpdir;
  if (char* tmpdir_env = getenv("TEST_TMPDIR"); tmpdir_env != nullptr) {
    test_tmpdir = std::string(tmpdir_env);
  } else {
    test_tmpdir = std::filesystem::temp_directory_path();
  }

  const auto* unit_test = ::testing::UnitTest::GetInstance();
  const auto* test_info = unit_test->current_test_info();
  std::filesystem::path test_file =
      test_tmpdir / llvm::formatv("{0}_{1}_{2}", test_info->test_suite_name(),
                                  test_info->name(), name)
                        .str();
  // Make debugging a bit easier by cleaning up any files from previous runs.
  // This is only necessary when not run in Bazel's test environment.
  std::filesystem::remove(test_file);
  if (std::filesystem::exists(test_file)) {
    return Error(
        llvm::formatv("Unable to remove an existing file: {0}", test_file));
  }

  std::error_code ec;
  llvm::raw_fd_ostream test_file_stream(test_file.string(), ec);
  if (ec) {
    return Error(llvm::formatv("Test file error: {0}", ec.message()));
  }
  test_file_stream << contents;

  return test_file;
}

}  // namespace Carbon::Testing
