// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TESTING_FILE_TEST_RUN_TEST_H_
#define CARBON_TESTING_FILE_TEST_RUN_TEST_H_

#include "common/error.h"
#include "testing/file_test/file_test_base.h"
#include "testing/file_test/test_file.h"

namespace Carbon::Testing {

// Runs the test, updating `test_file`.
auto RunTestFile(const FileTestBase& test_base, bool dump_output,
                 TestFile& test_file) -> ErrorOr<Success>;

}  // namespace Carbon::Testing

#endif  // CARBON_TESTING_FILE_TEST_RUN_TEST_H_
