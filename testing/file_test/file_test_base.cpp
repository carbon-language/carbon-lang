// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/file_test/file_test_base.h"

#include <filesystem>
#include <optional>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "common/check.h"
#include "common/error.h"
#include "common/exe_path.h"
#include "common/init_llvm.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/ThreadPool.h"
#include "testing/base/file_helpers.h"
#include "testing/file_test/autoupdate.h"
#include "testing/file_test/run_test.h"
#include "testing/file_test/test_file.h"

ABSL_FLAG(std::vector<std::string>, file_tests, {},
          "A comma-separated list of repo-relative names of test files. "
          "Overrides test_targets_file.");
ABSL_FLAG(std::string, test_targets_file, "",
          "A path to a file containing repo-relative names of test files.");
ABSL_FLAG(bool, autoupdate, false,
          "Instead of verifying files match test output, autoupdate files "
          "based on test output.");
ABSL_FLAG(unsigned int, threads, 0,
          "Number of threads to use when autoupdating tests, or 0 to "
          "automatically determine a thread count.");
ABSL_FLAG(bool, dump_output, false,
          "Instead of verifying files match test output, directly dump output "
          "to stderr.");

namespace Carbon::Testing {

// Splits outputs to string_view because gtest handles string_view by default.
static auto SplitOutput(llvm::StringRef output)
    -> llvm::SmallVector<std::string_view> {
  if (output.empty()) {
    return {};
  }
  llvm::SmallVector<llvm::StringRef> lines;
  llvm::StringRef(output).split(lines, "\n");
  return llvm::SmallVector<std::string_view>(lines.begin(), lines.end());
}

// Verify that the success and `fail_` prefix use correspond. Separately handle
// both cases for clearer test failures.
static auto CompareFailPrefix(llvm::StringRef filename, bool success) -> void {
  if (success) {
    EXPECT_FALSE(filename.starts_with("fail_"))
        << "`" << filename
        << "` succeeded; if success is expected, remove the `fail_` "
           "prefix.";
  } else {
    EXPECT_TRUE(filename.starts_with("fail_"))
        << "`" << filename
        << "` failed; if failure is expected, add the `fail_` prefix.";
  }
}

// Modes for GetBazelCommand.
enum class BazelMode : uint8_t {
  Autoupdate,
  Dump,
  Test,
};

// Returns the requested bazel command string for the given execution mode.
static auto GetBazelCommand(BazelMode mode, llvm::StringRef test_name)
    -> std::string {
  RawStringOstream args;

  const char* target = getenv("TEST_TARGET");
  args << "bazel " << ((mode == BazelMode::Test) ? "test" : "run") << " "
       << (target ? target : "<target>") << " ";

  switch (mode) {
    case BazelMode::Autoupdate:
      args << "-- --autoupdate ";
      break;

    case BazelMode::Dump:
      args << "-- --dump_output ";
      break;

    case BazelMode::Test:
      args << "--test_arg=";
      break;
  }

  args << "--file_tests=";
  args << test_name;
  return args.TakeStr();
}

// Runs the FileTestAutoupdater, returning the result.
static auto RunAutoupdater(FileTestBase* test_base, const TestFile& test_file,
                           bool dry_run) -> bool {
  if (!test_file.autoupdate_line_number) {
    return false;
  }

  llvm::SmallVector<llvm::StringRef> filenames;
  filenames.reserve(test_file.non_check_lines.size());
  if (test_file.has_splits) {
    // There are splits, so we provide an empty name for the first file.
    filenames.push_back({});
  }
  for (const auto& file : test_file.file_splits) {
    filenames.push_back(file.filename);
  }

  llvm::ArrayRef expected_filenames = filenames;
  if (filenames.size() > 1) {
    expected_filenames = expected_filenames.drop_front();
  }

  return FileTestAutoupdater(
             std::filesystem::absolute(test_base->test_name().str()),
             GetBazelCommand(BazelMode::Test, test_base->test_name()),
             GetBazelCommand(BazelMode::Dump, test_base->test_name()),
             test_file.input_content, filenames,
             *test_file.autoupdate_line_number, test_file.autoupdate_split,
             test_file.non_check_lines, test_file.actual_stdout,
             test_file.actual_stderr,
             test_base->GetDefaultFileRE(expected_filenames),
             test_base->GetLineNumberReplacements(expected_filenames),
             [&](std::string& line) {
               test_base->DoExtraCheckReplacements(line);
             })
      .Run(dry_run);
}

// Runs a test and compares output. This keeps output split by line so that
// issues are a little easier to identify by the different line.
auto FileTestBase::TestBody() -> void {
  // Add a crash trace entry with the single-file test command.
  std::string test_command = GetBazelCommand(BazelMode::Test, test_name_);
  llvm::PrettyStackTraceString stack_trace_entry(test_command.c_str());
  llvm::errs() << "\nTo test this file alone, run:\n  " << test_command
               << "\n\n";

  ErrorOr<TestFile> test_file =
      ProcessTestFileAndRun(this, output_mutex_, /*dump_output=*/false,
                            absl::GetFlag(FLAGS_autoupdate));
  ASSERT_TRUE(test_file.ok()) << test_file.error();
  auto test_filename = std::filesystem::path(test_name_.str()).filename();

  // Check success/failure against `fail_` prefixes.
  if (test_file->run_result.per_file_success.empty()) {
    CompareFailPrefix(test_filename.string(), test_file->run_result.success);
  } else {
    bool require_overall_failure = false;
    for (const auto& [filename, success] :
         test_file->run_result.per_file_success) {
      CompareFailPrefix(filename, success);
      if (!success) {
        require_overall_failure = true;
      }
    }

    if (require_overall_failure) {
      EXPECT_FALSE(test_file->run_result.success)
          << "There is a per-file failure expectation, so the overall result "
             "should have been a failure.";
    } else {
      // Individual files all succeeded, so the prefix is enforced on the main
      // test file.
      CompareFailPrefix(test_filename.string(), test_file->run_result.success);
    }
  }

  // Check results. Include a reminder of the autoupdate command for any
  // stdout/stderr differences.
  std::string update_message;
  if (test_file->autoupdate_line_number) {
    update_message = llvm::formatv(
        "If these differences are expected, try the autoupdater:\n  {0}",
        GetBazelCommand(BazelMode::Autoupdate, test_name_));
  } else {
    update_message =
        "If these differences are expected, content must be updated manually.";
  }
  SCOPED_TRACE(update_message);
  if (test_file->check_subset) {
    EXPECT_THAT(SplitOutput(test_file->actual_stdout),
                IsSupersetOf(test_file->expected_stdout));
    EXPECT_THAT(SplitOutput(test_file->actual_stderr),
                IsSupersetOf(test_file->expected_stderr));

  } else {
    EXPECT_THAT(SplitOutput(test_file->actual_stdout),
                ElementsAreArray(test_file->expected_stdout));
    EXPECT_THAT(SplitOutput(test_file->actual_stderr),
                ElementsAreArray(test_file->expected_stderr));
  }

  // If there are no other test failures, check if autoupdate would make
  // changes. We don't do this when there _are_ failures because the
  // SCOPED_TRACE already contains the autoupdate reminder.
  if (!HasFailure() && RunAutoupdater(this, *test_file, /*dry_run=*/true)) {
    ADD_FAILURE() << "Autoupdate would make changes to the file content.";
  }
}

auto FileTestBase::Autoupdate() -> ErrorOr<bool> {
  // Add a crash trace entry mentioning which file we're updating.
  std::string stack_trace_string =
      llvm::formatv("performing autoupdate for {0}", test_name_);
  llvm::PrettyStackTraceString stack_trace_entry(stack_trace_string.c_str());

  CARBON_ASSIGN_OR_RETURN(
      TestFile test_file,
      ProcessTestFileAndRun(this, output_mutex_, /*dump_output=*/false,
                            absl::GetFlag(FLAGS_autoupdate)));
  return RunAutoupdater(this, test_file, /*dry_run=*/false);
}

auto FileTestBase::DumpOutput() -> ErrorOr<Success> {
  std::string banner(79, '=');
  banner.append("\n");
  llvm::errs() << banner << "= " << test_name_ << "\n";

  CARBON_ASSIGN_OR_RETURN(
      TestFile test_file,
      ProcessTestFileAndRun(this, output_mutex_, /*dump_output=*/true,
                            absl::GetFlag(FLAGS_autoupdate)));
  llvm::errs() << banner << test_file.actual_stdout << banner
               << "= Exit with success: "
               << (test_file.run_result.success ? "true" : "false") << "\n"
               << banner;
  return Success();
}

auto FileTestBase::GetLineNumberReplacements(
    llvm::ArrayRef<llvm::StringRef> filenames)
    -> llvm::SmallVector<LineNumberReplacement> {
  return {{.has_file = true,
           .re = std::make_shared<RE2>(
               llvm::formatv(R"(({0}):(\d+)?)", llvm::join(filenames, "|"))),
           .line_formatv = R"({0})"}};
}

// Returns the tests to run.
static auto GetTests() -> llvm::SmallVector<std::string> {
  // Prefer a user-specified list if present.
  auto specific_tests = absl::GetFlag(FLAGS_file_tests);
  if (!specific_tests.empty()) {
    return llvm::SmallVector<std::string>(specific_tests.begin(),
                                          specific_tests.end());
  }

  // Extracts tests from the target file.
  CARBON_CHECK(!absl::GetFlag(FLAGS_test_targets_file).empty(),
               "Missing --test_targets_file.");
  auto content = ReadFile(absl::GetFlag(FLAGS_test_targets_file));
  CARBON_CHECK(content.ok(), "{0}", content.error());
  llvm::SmallVector<std::string> all_tests;
  for (llvm::StringRef file_ref : llvm::split(*content, "\n")) {
    if (file_ref.empty()) {
      continue;
    }
    all_tests.push_back(file_ref.str());
  }
  return all_tests;
}

// Runs autoupdate for the given tests. This is multi-threaded to try to get a
// little extra speed.
static auto RunAutoupdate(llvm::StringRef exe_path,
                          llvm::ArrayRef<std::string> tests,
                          FileTestFactory& test_factory) -> int {
  llvm::CrashRecoveryContext::Enable();
  llvm::DefaultThreadPool pool(
      {.ThreadsRequested = absl::GetFlag(FLAGS_threads)});

  // Guard access to both `llvm::errs` and `crashed`.
  std::mutex mutex;
  bool crashed = false;

  for (const auto& test_name : tests) {
    pool.async([&test_factory, &mutex, &exe_path, &crashed, test_name] {
      // If any thread crashed, don't try running more.
      {
        std::unique_lock<std::mutex> lock(mutex);
        if (crashed) {
          return;
        }
      }

      // Use a crash recovery context to try to get a stack trace when
      // multiple threads may crash in parallel, which otherwise leads to the
      // program aborting without printing a stack trace.
      llvm::CrashRecoveryContext crc;
      crc.DumpStackAndCleanupOnFailure = true;
      bool thread_crashed = !crc.RunSafely([&] {
        std::unique_ptr<FileTestBase> test(
            test_factory.factory_fn(exe_path, &mutex, test_name));
        auto result = test->Autoupdate();

        std::unique_lock<std::mutex> lock(mutex);
        if (result.ok()) {
          llvm::errs() << (*result ? "!" : ".");
        } else {
          llvm::errs() << "\n" << result.error().message() << "\n";
        }
      });
      if (thread_crashed) {
        std::unique_lock<std::mutex> lock(mutex);
        crashed = true;
      }
    });
  }

  pool.wait();
  if (crashed) {
    // Abort rather than returning so that we don't get a LeakSanitizer report.
    // We expect to have leaked memory if one or more of our tests crashed.
    std::abort();
  }
  llvm::errs() << "\nDone!\n";
  return EXIT_SUCCESS;
}

// Implements main() within the Carbon::Testing namespace for convenience.
static auto Main(int argc, char** argv) -> int {
  Carbon::InitLLVM init_llvm(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  auto args = absl::ParseCommandLine(argc, argv);

  if (args.size() > 1) {
    llvm::errs() << "Unexpected arguments:";
    for (char* arg : llvm::ArrayRef(args).drop_front()) {
      llvm::errs() << " ";
      llvm::errs().write_escaped(arg);
    }
    llvm::errs() << "\n";
    return EXIT_FAILURE;
  }

  std::string exe_path = FindExecutablePath(argv[0]);

  // Tests might try to read from stdin. Ensure those reads fail by closing
  // stdin and reopening it as /dev/null. Note that STDIN_FILENO doesn't exist
  // on Windows, but POSIX requires it to be 0.
  if (std::error_code error =
          llvm::sys::Process::SafelyCloseFileDescriptor(0)) {
    llvm::errs() << "Unable to close standard input: " << error.message()
                 << "\n";
    return EXIT_FAILURE;
  }
  if (std::error_code error =
          llvm::sys::Process::FixupStandardFileDescriptors()) {
    llvm::errs() << "Unable to correct standard file descriptors: "
                 << error.message() << "\n";
    return EXIT_FAILURE;
  }
  if (absl::GetFlag(FLAGS_autoupdate) && absl::GetFlag(FLAGS_dump_output)) {
    llvm::errs() << "--autoupdate and --dump_output are mutually exclusive.\n";
    return EXIT_FAILURE;
  }

  llvm::SmallVector<std::string> tests = GetTests();
  auto test_factory = GetFileTestFactory();
  if (absl::GetFlag(FLAGS_autoupdate)) {
    return RunAutoupdate(exe_path, tests, test_factory);
  } else if (absl::GetFlag(FLAGS_dump_output)) {
    for (const auto& test_name : tests) {
      std::unique_ptr<FileTestBase> test(
          test_factory.factory_fn(exe_path, nullptr, test_name));
      auto result = test->DumpOutput();
      if (!result.ok()) {
        llvm::errs() << "\n" << result.error().message() << "\n";
      }
    }
    llvm::errs() << "\nDone!\n";
    return EXIT_SUCCESS;
  } else {
    for (const std::string& test_name : tests) {
      testing::RegisterTest(
          test_factory.name, test_name.c_str(), nullptr, test_name.c_str(),
          __FILE__, __LINE__,
          [&test_factory, &exe_path, test_name = test_name]() {
            return test_factory.factory_fn(exe_path, nullptr, test_name);
          });
    }
    return RUN_ALL_TESTS();
  }
}

}  // namespace Carbon::Testing

auto main(int argc, char** argv) -> int {
  return Carbon::Testing::Main(argc, argv);
}
