// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TESTING_FILE_TEST_FILE_TEST_BASE_H_
#define CARBON_TESTING_FILE_TEST_FILE_TEST_BASE_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <functional>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon::Testing {

// A framework for testing files. Children implement `RegisterTestFiles` with
// calls to `RegisterTests` using a factory that constructs the child.
// `RunOverFile` must also be implemented and will be called as part of
// individual test executions. This framework includes a `main` implementation,
// so users must not provide one.
//
// Tests should have CHECK lines similar to `FileCheck` syntax:
//   https://llvm.org/docs/CommandGuide/FileCheck.html
//
// Special nuances are that stdout and stderr will look like `// CHECK:STDOUT:
// ...` and `// CHECK:STDERR: ...` respectively. `[[@LINE+offset]` and
// `{{regex}}` syntaxes should also work.
//
// `lit_autoupdate.py` automatically constructs compatible lines.
class FileTestBase : public testing::Test {
 public:
  explicit FileTestBase(const llvm::StringRef path) : path_(path) {}

  // Used by children to register tests with gtest.
  static void RegisterTests(
      const char* fixture_label, const std::vector<llvm::StringRef>& paths,
      std::function<FileTestBase*(llvm::StringRef)> factory);

  // Implemented by children to run the test. Called by the TestBody
  // implementation, which will validate stdout and stderr. The return value
  // should be false when "fail_" is in the filename.
  virtual auto RunOverFile(llvm::raw_ostream& stdout, llvm::raw_ostream& stderr)
      -> bool = 0;

  // Runs a test and compares output. This keeps output split by line so that
  // issues are a little easier to identify by the different line.
  auto TestBody() -> void final;

  // Returns the filename of the file being tested.
  auto filename() -> llvm::StringRef;

  // Returns the full path of the file being tested.
  auto path() -> llvm::StringRef { return path_; };

 private:
  // Transforms an expectation on a given line from `FileCheck` syntax into a
  // standard regex matcher.
  static auto TransformExpectation(int line_index, llvm::StringRef in)
      -> testing::Matcher<std::string>;

  llvm::StringRef path_;
};

// Must be implemented by the individual file_test to initialize tests.
extern auto RegisterFileTests(const std::vector<llvm::StringRef>& paths)
    -> void;

}  // namespace Carbon::Testing

#endif  // CARBON_TESTING_FILE_TEST_FILE_TEST_BASE_H_
