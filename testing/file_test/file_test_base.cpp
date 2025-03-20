// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implementation-wise, this:
//
// - Uses the registered `FileTestFactory` to construct `FileTestBase`
//   instances.
// - Constructs a `FileTestCase` that wraps each `FileTestBase` instance to
//   register with googletest, and to provide the actual `TestBody`.
// - Using `FileTestEventListener`, runs tests in parallel prior to normal
//   googletest execution.
//   - This is required to support `--gtest_filter` and access `should_run`.
//   - Runs each `FileTestBase` instance to cache the `TestFile` on
//     `FileTestInfo`.
//   - Determines whether autoupdate would make changes, autoupdating if
//     requested.
// - When googletest would normally execute the test, `FileTestCase::TestBody`
//   instead uses the cached state on `FileTestInfo`.
//   - This only occurs when neither autoupdating nor dumping output.

#include "testing/file_test/file_test_base.h"

#include <filesystem>
#include <optional>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_join.h"
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
#include "testing/file_test/autoupdate.h"
#include "testing/file_test/run_test.h"
#include "testing/file_test/test_file.h"

ABSL_FLAG(std::vector<std::string>, file_tests, {},
          "A comma-separated list of repo-relative names of test files. "
          "Similar to and overrides `--gtest_filter`, but doesn't require the "
          "test class name to be known.");
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

// Information for a test case.
struct FileTestInfo {
  // The name.
  std::string test_name;

  // A factory function for creating the test object.
  std::function<auto()->FileTestBase*> factory_fn;

  // gtest's information about the test.
  ::testing::TestInfo* registered_test;

  // The test result, set after running.
  std::optional<ErrorOr<TestFile>> test_result;

  // Whether running autoupdate would change (or when autoupdating, already
  // changed) the test file. This may be true even if output passes test
  // expectations.
  bool autoupdate_differs = false;
};

// Adapts a `FileTestBase` instance to gtest for outputting results.
class FileTestCase : public testing::Test {
 public:
  explicit FileTestCase(FileTestInfo* test_info) : test_info_(test_info) {}

  // Runs a test and compares output. This keeps output split by line so that
  // issues are a little easier to identify by the different line.
  auto TestBody() -> void final;

 private:
  FileTestInfo* test_info_;
};

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

auto FileTestCase::TestBody() -> void {
  if (absl::GetFlag(FLAGS_autoupdate) || absl::GetFlag(FLAGS_dump_output)) {
    return;
  }

  CARBON_CHECK(test_info_->test_result,
               "Expected test to be run prior to TestBody: {0}",
               test_info_->test_name);

  ASSERT_TRUE(test_info_->test_result->ok())
      << test_info_->test_result->error();
  auto test_filename = std::filesystem::path(test_info_->test_name).filename();

  // Check success/failure against `fail_` prefixes.
  TestFile& test_file = **(test_info_->test_result);
  if (test_file.run_result.per_file_success.empty()) {
    CompareFailPrefix(test_filename.string(), test_file.run_result.success);
  } else {
    bool require_overall_failure = false;
    for (const auto& [filename, success] :
         test_file.run_result.per_file_success) {
      CompareFailPrefix(filename, success);
      if (!success) {
        require_overall_failure = true;
      }
    }

    if (require_overall_failure) {
      EXPECT_FALSE(test_file.run_result.success)
          << "There is a per-file failure expectation, so the overall result "
             "should have been a failure.";
    } else {
      // Individual files all succeeded, so the prefix is enforced on the main
      // test file.
      CompareFailPrefix(test_filename.string(), test_file.run_result.success);
    }
  }

  // Check results. Include a reminder for NOAUTOUPDATE tests.
  std::unique_ptr<testing::ScopedTrace> scoped_trace;
  if (!test_file.autoupdate_line_number) {
    scoped_trace = std::make_unique<testing::ScopedTrace>(
        __FILE__, __LINE__,
        "This file is NOAUTOUPDATE, so expected differences require manual "
        "updates.");
  }
  if (test_file.check_subset) {
    EXPECT_THAT(SplitOutput(test_file.actual_stdout),
                IsSupersetOf(test_file.expected_stdout));
    EXPECT_THAT(SplitOutput(test_file.actual_stderr),
                IsSupersetOf(test_file.expected_stderr));

  } else {
    EXPECT_THAT(SplitOutput(test_file.actual_stdout),
                ElementsAreArray(test_file.expected_stdout));
    EXPECT_THAT(SplitOutput(test_file.actual_stderr),
                ElementsAreArray(test_file.expected_stderr));
  }

  if (HasFailure()) {
    llvm::errs() << "\nTo test this file alone, run:\n  "
                 << GetBazelCommand(BazelMode::Test, test_info_->test_name)
                 << "\n\n";
    if (!test_file.autoupdate_line_number) {
      llvm::errs() << "\nThis test is NOAUTOUPDATE.\n\n";
    }
  }
  if (test_info_->autoupdate_differs) {
    ADD_FAILURE() << "Autoupdate would make changes to the file content. Run:\n"
                  << GetBazelCommand(BazelMode::Autoupdate,
                                     test_info_->test_name);
  }
}

auto FileTestBase::GetLineNumberReplacements(
    llvm::ArrayRef<llvm::StringRef> filenames) const
    -> llvm::SmallVector<LineNumberReplacement> {
  return {{.has_file = true,
           .re = std::make_shared<RE2>(
               llvm::formatv(R"(({0}):(\d+)?)", llvm::join(filenames, "|"))),
           .line_formatv = R"({0})"}};
}

// If `--file_tests` is set, transform it into a `--gtest_filter`.
static auto MaybeApplyFileTestsFlag(llvm::StringRef factory_name) -> void {
  if (absl::GetFlag(FLAGS_file_tests).empty()) {
    return;
  }
  RawStringOstream filter;
  llvm::ListSeparator sep(":");
  for (const auto& file : absl::GetFlag(FLAGS_file_tests)) {
    filter << sep << factory_name << "." << file;
  }
  absl::SetFlag(&FLAGS_gtest_filter, filter.TakeStr());
}

// Loads tests from the manifest file, and registers them for execution. The
// vector is taken as an output parameter so that the address of entries is
// stable for the factory.
static auto RegisterTests(FileTestFactory* test_factory,
                          llvm::StringRef exe_path,
                          llvm::SmallVectorImpl<FileTestInfo>& tests)
    -> ErrorOr<Success> {
  // Prepare the vector first, so that the location of entries won't change.
  for (auto& test_name : GetFileTestManifest()) {
    tests.push_back({.test_name = test_name});
  }

  // Amend entries with factory functions.
  for (auto& test : tests) {
    llvm::StringRef test_name = test.test_name;
    test.factory_fn = [test_factory, exe_path, test_name]() {
      return test_factory->factory_fn(exe_path, test_name);
    };
    test.registered_test = testing::RegisterTest(
        test_factory->name, test_name.data(), nullptr, test_name.data(),
        __FILE__, __LINE__, [&test]() { return new FileTestCase(&test); });
  }
  return Success();
}

// Implements the parallel test execution through gtest's listener support.
class FileTestEventListener : public testing::EmptyTestEventListener {
 public:
  explicit FileTestEventListener(llvm::MutableArrayRef<FileTestInfo> tests)
      : tests_(tests) {}

  // Runs test during start, after `should_run` is initialized. This is
  // multi-threaded to get extra speed.
  auto OnTestProgramStart(const testing::UnitTest& /*unit_test*/)
      -> void override;

 private:
  llvm::MutableArrayRef<FileTestInfo> tests_;
};

// Returns true if the main thread should be used to run tests. This is if
// either --dump_output is specified, or only 1 thread is needed to run tests.
static auto SingleThreaded(llvm::ArrayRef<FileTestInfo> tests) -> bool {
  if (absl::GetFlag(FLAGS_dump_output) || absl::GetFlag(FLAGS_threads) == 1) {
    return true;
  }

  bool found_test_to_run = false;
  for (const auto& test : tests) {
    if (!test.registered_test->should_run()) {
      continue;
    }
    if (found_test_to_run) {
      // At least two tests will run, so multi-threaded.
      return false;
    }
    // Found the first test to run.
    found_test_to_run = true;
  }
  // 0 or 1 test will be run, so single-threaded.
  return false;
}

// Runs the test in the section that would be inside a lock, possibly inside a
// CrashRecoveryContext.
static auto RunSingleTestHelper(FileTestInfo& test, FileTestBase& test_instance)
    -> void {
  // Add a crash trace entry with the single-file test command.
  std::string test_command = GetBazelCommand(BazelMode::Test, test.test_name);
  llvm::PrettyStackTraceString stack_trace_entry(test_command.c_str());

  if (auto err = RunTestFile(test_instance, absl::GetFlag(FLAGS_dump_output),
                             **test.test_result);
      !err.ok()) {
    test.test_result = std::move(err).error();
  }
}

// Runs a single test. Uses a CrashRecoveryContext, and returns false on a
// crash.
static auto RunSingleTest(FileTestInfo& test, bool single_threaded,
                          std::mutex& output_mutex) -> bool {
  std::unique_ptr<FileTestBase> test_instance(test.factory_fn());

  if (absl::GetFlag(FLAGS_dump_output)) {
    std::unique_lock<std::mutex> lock(output_mutex);
    llvm::errs() << "\n--- Dumping: " << test.test_name << "\n\n";
  }

  // Load expected output.
  test.test_result = ProcessTestFile(test_instance->test_name(),
                                     absl::GetFlag(FLAGS_autoupdate));
  if (test.test_result->ok()) {
    // Execution must be serialized for either serial tests or console
    // output.
    std::unique_lock<std::mutex> output_lock;

    if ((*test.test_result)->capture_console_output ||
        !test_instance->AllowParallelRun()) {
      output_lock = std::unique_lock<std::mutex>(output_mutex);
    }

    if (single_threaded) {
      RunSingleTestHelper(test, *test_instance);
    } else {
      // Use a crash recovery context to try to get a stack trace when
      // multiple threads may crash in parallel, which otherwise leads to the
      // program aborting without printing a stack trace.
      llvm::CrashRecoveryContext crc;
      crc.DumpStackAndCleanupOnFailure = true;
      if (!crc.RunSafely([&] { RunSingleTestHelper(test, *test_instance); })) {
        return false;
      }
    }
  }

  if (!test.test_result->ok()) {
    std::unique_lock<std::mutex> lock(output_mutex);
    llvm::errs() << "\n" << test.test_result->error().message() << "\n";
    return true;
  }

  test.autoupdate_differs =
      RunAutoupdater(test_instance.get(), **test.test_result,
                     /*dry_run=*/!absl::GetFlag(FLAGS_autoupdate));

  std::unique_lock<std::mutex> lock(output_mutex);
  if (absl::GetFlag(FLAGS_dump_output)) {
    llvm::outs().flush();
    const TestFile& test_file = **test.test_result;
    llvm::errs() << "\n--- Exit with success: "
                 << (test_file.run_result.success ? "true" : "false")
                 << "\n--- Autoupdate differs: "
                 << (test.autoupdate_differs ? "true" : "false") << "\n";
  } else {
    llvm::errs() << (test.autoupdate_differs ? "!" : ".");
  }

  return true;
}

auto FileTestEventListener::OnTestProgramStart(
    const testing::UnitTest& /*unit_test*/) -> void {
  bool single_threaded = SingleThreaded(tests_);

  std::unique_ptr<llvm::ThreadPoolInterface> pool;
  if (single_threaded) {
    pool = std::make_unique<llvm::SingleThreadExecutor>();
  } else {
    // Enable the CRC for use in `RunSingleTest`.
    llvm::CrashRecoveryContext::Enable();
    pool = std::make_unique<llvm::DefaultThreadPool>(llvm::ThreadPoolStrategy{
        .ThreadsRequested = absl::GetFlag(FLAGS_threads)});
  }
  if (!absl::GetFlag(FLAGS_dump_output)) {
    llvm::errs() << "Running tests with " << pool->getMaxConcurrency()
                 << " thread(s)\n";
  }

  // Guard access to output (stdout and stderr).
  std::mutex output_mutex;
  std::atomic<bool> crashed = false;

  for (auto& test : tests_) {
    if (!test.registered_test->should_run()) {
      continue;
    }

    pool->async([&] {
      // If any thread crashed, don't try running more.
      if (crashed) {
        return;
      }

      if (!RunSingleTest(test, single_threaded, output_mutex)) {
        crashed = true;
      }
    });
  }

  pool->wait();
  if (crashed) {
    // Abort rather than returning so that we don't get a LeakSanitizer report.
    // We expect to have leaked memory if one or more of our tests crashed.
    std::abort();
  }
  llvm::errs() << "\nDone!\n";
}

// Implements main() within the Carbon::Testing namespace for convenience.
static auto Main(int argc, char** argv) -> ErrorOr<int> {
  Carbon::InitLLVM init_llvm(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  auto args = absl::ParseCommandLine(argc, argv);

  if (args.size() > 1) {
    ErrorBuilder b;
    b << "Unexpected arguments:";
    for (char* arg : llvm::ArrayRef(args).drop_front()) {
      b << " " << FormatEscaped(arg);
    }
    return b;
  }

  std::string exe_path = FindExecutablePath(argv[0]);

  // Tests might try to read from stdin. Ensure those reads fail by closing
  // stdin and reopening it as /dev/null. Note that STDIN_FILENO doesn't exist
  // on Windows, but POSIX requires it to be 0.
  if (std::error_code error =
          llvm::sys::Process::SafelyCloseFileDescriptor(0)) {
    return Error("Unable to close standard input: " + error.message());
  }
  if (std::error_code error =
          llvm::sys::Process::FixupStandardFileDescriptors()) {
    return Error("Unable to correct standard file descriptors: " +
                 error.message());
  }
  if (absl::GetFlag(FLAGS_autoupdate) && absl::GetFlag(FLAGS_dump_output)) {
    return Error("--autoupdate and --dump_output are mutually exclusive.");
  }

  auto test_factory = GetFileTestFactory();

  MaybeApplyFileTestsFlag(test_factory.name);

  // Inline 0 entries because it will always be too large to store on the stack.
  llvm::SmallVector<FileTestInfo, 0> tests;
  CARBON_RETURN_IF_ERROR(RegisterTests(&test_factory, exe_path, tests));

  testing::TestEventListeners& listeners =
      testing::UnitTest::GetInstance()->listeners();
  if (absl::GetFlag(FLAGS_autoupdate) || absl::GetFlag(FLAGS_dump_output)) {
    // Suppress all of the default output.
    delete listeners.Release(listeners.default_result_printer());
  }
  // Use a listener to run tests in parallel.
  listeners.Append(new FileTestEventListener(tests));

  return RUN_ALL_TESTS();
}

}  // namespace Carbon::Testing

auto main(int argc, char** argv) -> int {
  if (auto result = Carbon::Testing::Main(argc, argv); result.ok()) {
    return *result;
  } else {
    llvm::errs() << result.error() << "\n";
    return EXIT_FAILURE;
  }
}
