// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TESTING_BASE_FILE_HELPERS_H_
#define CARBON_TESTING_BASE_FILE_HELPERS_H_

#include <filesystem>
#include <string>

#include "common/error.h"

namespace Carbon::Testing {

// Reads a file to string.
auto ReadFile(std::filesystem::path path) -> ErrorOr<std::string>;

// Writes a test file to disk and returns an error or the full path to the file.
//
// Note that this expects to be run from within a GoogleTest test, and relies on
// global state of GoogleTest.
//
// This locates a suitable temporary directory for the test, creates a file with
// the requested name in that directory, and writes the provided content to that
// file. The full path to the written file is returned.
//
// Where possible, this will use a Bazel-provided test temporary directory.
// However, if unavailable, it falls back to a system temporary directory. This
// helps tests be runnable outside of Bazel, for example under a debugger. It
// also works to create test file names that are unlikely to conflict with other
// tests when run.
auto WriteTestFile(llvm::StringRef name, llvm::StringRef contents)
    -> ErrorOr<std::filesystem::path>;

}  // namespace Carbon::Testing

#endif  // CARBON_TESTING_BASE_FILE_HELPERS_H_
