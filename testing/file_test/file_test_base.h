// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TESTING_FILE_TEST_FILE_TEST_BASE_H_
#define CARBON_TESTING_FILE_TEST_FILE_TEST_BASE_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <functional>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon::Testing {

// A framework for testing files. Children implement `RegisterTestFiles` with
// calls to `RegisterTests` using a factory that constructs the child.
// `RunWithFiles` must also be implemented and will be called as part of
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
// `autoupdate_testdata.py` automatically constructs compatible lines.
class FileTestBase : public testing::Test {
 public:
  struct TestFile {
    explicit TestFile(std::string filename, llvm::StringRef content)
        : filename(std::move(filename)), content(content) {}

    friend void PrintTo(const TestFile& f, std::ostream* os) {
      // Print content escaped.
      llvm::raw_os_ostream os_wrap(*os);
      os_wrap << "TestFile(" << f.filename << ", \"";
      os_wrap.write_escaped(f.content);
      os_wrap << "\")";
    }

    std::string filename;
    llvm::StringRef content;
  };

  explicit FileTestBase(const std::filesystem::path& path) : path_(&path) {}

  // Used by children to register tests with gtest.
  static auto RegisterTests(
      const char* fixture_label,
      const llvm::SmallVector<std::filesystem::path>& paths,
      std::function<FileTestBase*(const std::filesystem::path&)> factory)
      -> void;

  template <typename FileTestChildT>
  static auto RegisterTests(
      const char* fixture_label,
      const llvm::SmallVector<std::filesystem::path>& paths) -> void {
    RegisterTests(fixture_label, paths, [](const std::filesystem::path& path) {
      return new FileTestChildT(path);
    });
  }

  // Implemented by children to run the test. Called by the TestBody
  // implementation, which will validate stdout and stderr. The return value
  // should be false when "fail_" is in the filename.
  virtual auto RunWithFiles(const llvm::SmallVector<TestFile>& test_files,
                            llvm::raw_pwrite_stream& stdout,
                            llvm::raw_pwrite_stream& stderr) -> bool = 0;

  // Runs a test and compares output. This keeps output split by line so that
  // issues are a little easier to identify by the different line.
  auto TestBody() -> void final;

  // Returns the full path of the file being tested.
  auto path() -> const std::filesystem::path& { return *path_; };

 private:
  // Processes the test input, producing test files and expected output.
  auto ProcessTestFile(
      llvm::StringRef file_content, llvm::SmallVector<TestFile>& test_files,
      llvm::SmallVector<testing::Matcher<std::string>>& expected_stdout,
      llvm::SmallVector<testing::Matcher<std::string>>& expected_stderr)
      -> void;

  // Transforms an expectation on a given line from `FileCheck` syntax into a
  // standard regex matcher.
  static auto TransformExpectation(int line_index, llvm::StringRef in)
      -> testing::Matcher<std::string>;

  const std::filesystem::path* path_;
};

// Must be implemented by the individual file_test to initialize tests.
extern auto RegisterFileTests(
    const llvm::SmallVector<std::filesystem::path>& paths) -> void;

}  // namespace Carbon::Testing

#endif  // CARBON_TESTING_FILE_TEST_FILE_TEST_BASE_H_
