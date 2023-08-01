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

// A framework for testing files. Children write
// `CARBON_FILE_TEST_FACTORY(MyTest)` which is used to construct the tests.
// `RunWithFiles` must also be implemented and will be called as part of
// individual test executions. This framework includes a `main` implementation,
// so users must not provide one.
//
// Settings in files are provided in comments, similar to `FileCheck` syntax.
// `autoupdate_testdata.py` automatically constructs compatible CHECK:STDOUT:
// and CHECK:STDERR: lines.
//
// Supported comment markers are:
//
// - // ARGS: <arguments>
//
//   Provides a space-separated list of arguments, which will be passed to
//   RunWithFiles as test_args. These are intended for use by the command as
//   arguments.
//
//   Supported replacements within arguments are:
//
//   - %s
//
//     Replaced with the list of files. Currently only allowed as a standalone
//     argument, not a substring.
//
//   - %t
//
//     Replaced with `${TEST_TMPDIR}/temp_file`.
//
//   ARGS can be specified at most once. If not provided, the FileTestBase child
//   is responsible for providing default arguments.
//
// - // SET-CHECK-SUBSET
//
//   By default, all lines of output must have a CHECK match. Adding this as a
//   flag sets it so that non-matching lines are ignored. All provided
//   CHECK:STDOUT: and CHECK:STDERR: lines must still have a match in output.
//
//   SET-CHECK-SUBSET can be specified at most once.
//
// - // --- <filename>
//
//   By default, all file content is provided to the test as a single file in
//   test_files. Using this marker allows the file to be split into multiple
//   files which will all be passed to test_files.
//
//   Files are not created on disk; it's expected the child will create an
//   InMemoryFilesystem if needed.
//
// - // CHECK:STDOUT: <output line>
//   // CHECK:STDERR: <output line>
//
//   These provides a match for output from the command. See SET-CHECK-SUBSET
//   for how to change from full to subset matching of output.
//
//   Output line matchers may contain `[[@LINE+offset]` and
//   `{{regex}}` syntaxes, similar to `FileCheck`.
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

  explicit FileTestBase(std::filesystem::path path) : path_(std::move(path)) {}

  // Implemented by children to run the test. Called by the TestBody
  // implementation, which will validate stdout and stderr. The return value
  // should be false when "fail_" is in the filename.
  virtual auto RunWithFiles(const llvm::SmallVector<llvm::StringRef>& test_args,
                            const llvm::SmallVector<TestFile>& test_files,
                            llvm::raw_pwrite_stream& stdout,
                            llvm::raw_pwrite_stream& stderr) -> bool = 0;

  // Returns default arguments. Only called when a file doesn't set ARGS.
  virtual auto GetDefaultArgs() -> llvm::SmallVector<std::string> = 0;

  // Runs a test and compares output. This keeps output split by line so that
  // issues are a little easier to identify by the different line.
  auto TestBody() -> void final;

  // Returns the full path of the file being tested.
  auto path() -> const std::filesystem::path& { return path_; };

 private:
  // Does replacements in ARGS for %s and %t.
  auto DoArgReplacements(llvm::SmallVector<std::string>& test_args,
                         const llvm::SmallVector<TestFile>& test_files) -> void;

  // Processes the test input, producing test files and expected output.
  auto ProcessTestFile(
      llvm::StringRef file_content, llvm::SmallVector<std::string>& test_args,
      llvm::SmallVector<TestFile>& test_files,
      llvm::SmallVector<testing::Matcher<std::string>>& expected_stdout,
      llvm::SmallVector<testing::Matcher<std::string>>& expected_stderr,
      bool& check_subset) -> void;

  // Transforms an expectation on a given line from `FileCheck` syntax into a
  // standard regex matcher.
  static auto TransformExpectation(int line_index, llvm::StringRef in)
      -> testing::Matcher<std::string>;

  const std::filesystem::path path_;
};
// A plain struct to aggregate a name and factory function for tests using this framework.
struct FileTestFactory {
  // The test fixture name.
  const char* name;
  // A factory function for tests.
  std::function<FileTestBase*(const std::filesystem::path& path)> factory_fn;
};

// Must be implemented by the individual file_test to initialize tests.
//
// We can't use INSTANTIATE_TEST_CASE_P because of ordering issues between
// container initialization and test instantiation by InitGoogleTest, but this
// also allows us more flexibility in execution.
//
// We provide a `CARBON_FILE_TEST_FACTOR` macro below that provides a common,
// convenient way to implement this function.
extern auto GetFileTestFactory() -> FileTestFactory;

// Provides a standard way to implement GetFileTestFactory.
#define CARBON_FILE_TEST_FACTORY(Name)                                         \
  auto GetFileTestFactory()->FileTestFactory {                                 \
    return {(#Name),                                                           \
            [](const std::filesystem::path& path) { return new Name(path); }}; \
  }

}  // namespace Carbon::Testing

#endif  // CARBON_TESTING_FILE_TEST_FILE_TEST_BASE_H_
