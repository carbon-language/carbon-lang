// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TESTING_FILE_TEST_FILE_TEST_BASE_H_
#define CARBON_TESTING_FILE_TEST_FILE_TEST_BASE_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <functional>

#include "llvm/Support/raw_ostream.h"

namespace Carbon::Testing {

class FileTestBase : public testing::Test {
 public:
  explicit FileTestBase(const llvm::StringRef path) : path_(path) {}

  static void RegisterTests(
      const char* fixture_label, int argc, char** argv,
      std::function<FileTestBase*(llvm::StringRef)> factory);

  virtual auto RunOverFile(llvm::raw_ostream& stdout, llvm::raw_ostream& stderr)
      -> void = 0;

  // Runs a test and compares output. This keeps output split by line so that
  // issues are a little easier to identify by the different line.
  auto TestBody() -> void override;

  // Returns the filename for the given path.
  auto filename() -> llvm::StringRef;

  auto path() -> llvm::StringRef { return path_; };

  // Transforms an expectation on a given line from `FileCheck` syntax into a
  // standard regex matcher.
  static auto TransformExpectation(int line_index, llvm::StringRef in)
      -> testing::Matcher<std::string>;

 private:
  llvm::StringRef path_;
};

}  // namespace Carbon::Testing

#endif  // CARBON_TESTING_FILE_TEST_FILE_TEST_BASE_H_
