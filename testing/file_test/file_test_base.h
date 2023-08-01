// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TESTING_FILE_TEST_FILE_TEST_BASE_H_
#define CARBON_TESTING_FILE_TEST_FILE_TEST_BASE_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <functional>

#include "common/error.h"
#include "common/ostream.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "testing/file_test/autoupdate.h"

namespace Carbon::Testing {

// A framework for testing files. Children write
// `CARBON_FILE_TEST_FACTORY(MyTest)` which is used to construct the tests.
// `Run` must also be implemented and will be called as part of individual test
// executions. This framework includes a `main` implementation, so users must
// not provide one.
//
// Settings in files are provided in comments, similar to `FileCheck` syntax.
// `bazel run :file_test -- --autoupdate` automatically constructs compatible
// CHECK:STDOUT: and CHECK:STDERR: lines.
//
// Supported comment markers are:
//
// - // AUTOUDPATE
//   // NOAUTOUPDATE
//
//   Controls whether the checks in the file will be autoupdated if --autoupdate
//   is passed. Exactly one of these two markers must be present. If the file
//   uses splits, AUTOUPDATE must currently be before any splits.
//
//   When autoupdating, CHECKs will be inserted starting below AUTOUPDATE. When
//   a CHECK has line information, autoupdate will try to insert the CHECK
//   immediately above the line it's associated with. When that happens, any
//   following CHECK lines without line information will immediately follow,
//   between the CHECK with line information and the associated line.
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

  // Provided for child class convenience.
  using LineNumberReplacement = FileTestLineNumberReplacement;

  explicit FileTestBase(std::filesystem::path path) : path_(std::move(path)) {}

  // Implemented by children to run the test. For example, TestBody validates
  // stdout and stderr.
  //
  // Any test expectations should be called from ValidateRun, not Run.
  //
  // The return value should be an error if there was an abnormal error. It
  // should be true if a binary would return EXIT_SUCCESS, and false for
  // EXIT_FAILURE (which is a test success for `fail_*` tests).
  virtual auto Run(const llvm::SmallVector<llvm::StringRef>& test_args,
                   const llvm::SmallVector<TestFile>& test_files,
                   llvm::raw_pwrite_stream& stdout,
                   llvm::raw_pwrite_stream& stderr) -> ErrorOr<bool> = 0;

  // Implemented by children to do post-Run test expectations. Only called when
  // testing. Does not need to be provided if only CHECK test expectations are
  // used.
  virtual auto ValidateRun(const llvm::SmallVector<TestFile>& /*test_files*/)
      -> void {}

  // Returns default arguments. Only called when a file doesn't set ARGS.
  virtual auto GetDefaultArgs() -> llvm::SmallVector<std::string> = 0;

  // Returns replacement information for line numbers. See LineReplacement for
  // construction.
  virtual auto GetLineNumberReplacement(
      llvm::ArrayRef<llvm::StringRef> filenames) -> LineNumberReplacement;

  // Optionally allows children to provide extra replacements for autoupdate.
  virtual auto DoExtraCheckReplacements(std::string& /*check_line*/) -> void {}

  // Runs a test and compares output. This keeps output split by line so that
  // issues are a little easier to identify by the different line.
  auto TestBody() -> void final;

  // Runs the test and autoupdates checks. Returns true if updated.
  auto Autoupdate() -> bool;

  // Returns the full path of the file being tested.
  auto path() -> const std::filesystem::path& { return path_; };

 private:
  // Encapsulates test context generated by processing and running.
  struct TestContext {
    // The input test file content. Other parts may reference this.
    std::string input_content;

    // Lines which don't contain CHECKs, and thus need to be retained by
    // autoupdate. Their line number in the file is attached.
    //
    // If there are splits, then the line is in the respective file. For N
    // splits, there will be one vector for the parts of the input file which
    // are not in any split, plus one vector per split file.
    llvm::SmallVector<llvm::SmallVector<FileTestLine>> non_check_lines;

    // Arguments for the test, generated from ARGS.
    llvm::SmallVector<std::string> test_args;

    // Files in the test, generated by content and splits.
    llvm::SmallVector<TestFile> test_files;

    // The location of the autoupdate marker, for autoupdated files.
    std::optional<int> autoupdate_line_number;

    // Whether checks are a subset, generated from SET-CHECK-SUBSET.
    bool check_subset = false;

    // stdout and stderr based on CHECK lines in the file.
    llvm::SmallVector<testing::Matcher<std::string>> expected_stdout;
    llvm::SmallVector<testing::Matcher<std::string>> expected_stderr;

    // stdout and stderr from Run. 16 is arbitrary but a required value.
    llvm::SmallString<16> stdout;
    llvm::SmallString<16> stderr;

    // Whether Run exited with success.
    bool exit_with_success = false;
  };

  // Processes the test file and runs the test. Returns an error if something
  // went wrong.
  auto ProcessTestFileAndRun(TestContext& context) -> ErrorOr<Success>;

  // Does replacements in ARGS for %s and %t.
  auto DoArgReplacements(llvm::SmallVector<std::string>& test_args,
                         const llvm::SmallVector<TestFile>& test_files)
      -> ErrorOr<Success>;

  // Processes the test input, producing test files and expected output.
  auto ProcessTestFile(TestContext& context) -> ErrorOr<Success>;

  // Transforms an expectation on a given line from `FileCheck` syntax into a
  // standard regex matcher.
  static auto TransformExpectation(int line_index, llvm::StringRef in)
      -> testing::Matcher<std::string>;

  const std::filesystem::path path_;
};

// Aggregate a name and factory function for tests using this framework.
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
// The `CARBON_FILE_TEST_FACTOR` macro below provides a standard, convenient way
// to implement this function.
extern auto GetFileTestFactory() -> FileTestFactory;

// Provides a standard GetFileTestFactory implementation.
#define CARBON_FILE_TEST_FACTORY(Name)                                         \
  auto GetFileTestFactory()->FileTestFactory {                                 \
    return {(#Name),                                                           \
            [](const std::filesystem::path& path) { return new Name(path); }}; \
  }

}  // namespace Carbon::Testing

#endif  // CARBON_TESTING_FILE_TEST_FILE_TEST_BASE_H_
