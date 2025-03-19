// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/file_test/run_test.h"

#include <gtest/gtest.h>

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "testing/file_test/file_system.h"
#include "testing/file_test/file_test_base.h"
#include "testing/file_test/test_file.h"

namespace Carbon::Testing {

// While these are marked as "internal" APIs, they seem to work and be pretty
// widely used for their exact documented behavior.
using ::testing::internal::CaptureStderr;
using ::testing::internal::CaptureStdout;
using ::testing::internal::GetCapturedStderr;
using ::testing::internal::GetCapturedStdout;

static constexpr llvm::StringLiteral StdinFilename = "STDIN";

// Does replacements in ARGS for %s and %t.
static auto DoArgReplacements(
    llvm::SmallVector<std::string>& test_args,
    const llvm::StringMap<std::string>& replacements,
    const llvm::SmallVector<TestFile::Split>& split_files) -> ErrorOr<Success> {
  for (auto* it = test_args.begin(); it != test_args.end(); ++it) {
    auto percent = it->find("%");
    if (percent == std::string::npos) {
      continue;
    }

    if (percent + 1 >= it->size()) {
      return ErrorBuilder() << "% is not allowed on its own: " << *it;
    }
    char c = (*it)[percent + 1];
    switch (c) {
      case 's': {
        if (*it != "%s") {
          return ErrorBuilder() << "%s must be the full argument: " << *it;
        }
        it = test_args.erase(it);
        for (const auto& split : split_files) {
          const std::string& filename = split.filename;
          if (filename == StdinFilename || filename.ends_with(".h")) {
            continue;
          }
          it = test_args.insert(it, filename);
          ++it;
        }
        // Back up once because the for loop will advance.
        --it;
        break;
      }
      case 't': {
        char* tmpdir = getenv("TEST_TMPDIR");
        CARBON_CHECK(tmpdir != nullptr);
        it->replace(percent, 2, llvm::formatv("{0}/temp_file", tmpdir));
        break;
      }
      case '{': {
        auto end_brace = it->find('}', percent);
        if (end_brace == std::string::npos) {
          return ErrorBuilder() << "%{ without closing }: " << *it;
        }
        llvm::StringRef substr(&*(it->begin() + percent + 2),
                               end_brace - percent - 2);
        auto replacement = replacements.find(substr);
        if (replacement == replacements.end()) {
          return ErrorBuilder()
                 << "unknown substitution: %{" << substr << "}: " << *it;
        }
        it->replace(percent, end_brace - percent + 1, replacement->second);
        break;
      }
      default:
        return ErrorBuilder() << "%" << c << " is not supported: " << *it;
    }
  }
  return Success();
}

auto RunTestFile(const FileTestBase& test_base, bool dump_output,
                 TestFile& test_file) -> ErrorOr<Success> {
  // Process arguments.
  if (test_file.test_args.empty()) {
    test_file.test_args = test_base.GetDefaultArgs();
    test_file.test_args.append(test_file.extra_args);
  }
  for (const auto& include_file : test_file.include_files) {
    test_file.test_args.push_back(include_file);
  }
  CARBON_RETURN_IF_ERROR(DoArgReplacements(test_file.test_args,
                                           test_base.GetArgReplacements(),
                                           test_file.file_splits));

  // stdin needs to exist on-disk for compatibility. We'll use a pointer for it.
  FILE* input_stream = nullptr;
  auto erase_input_on_exit = llvm::make_scope_exit([&input_stream]() {
    if (input_stream) {
      // fclose should delete the tmpfile.
      fclose(input_stream);
      input_stream = nullptr;
    }
  });

  // Create the files in-memory.
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> fs =
      new llvm::vfs::InMemoryFileSystem;
  for (const auto& split : test_file.file_splits) {
    if (split.filename == StdinFilename) {
      input_stream = tmpfile();
      fwrite(split.content.c_str(), sizeof(char), split.content.size(),
             input_stream);
      rewind(input_stream);
    } else if (!fs->addFile(split.filename, /*ModificationTime=*/0,
                            llvm::MemoryBuffer::getMemBuffer(
                                split.content, split.filename,
                                /*RequiresNullTerminator=*/false))) {
      return ErrorBuilder() << "File is repeated: " << split.filename;
    }
  }

  for (llvm::StringRef file_path : test_file.include_files) {
    CARBON_RETURN_IF_ERROR(AddFile(*fs, file_path));
  }

  // Convert the arguments to StringRef and const char* to match the
  // expectations of PrettyStackTraceProgram and Run.
  llvm::SmallVector<llvm::StringRef> test_args_ref;
  llvm::SmallVector<const char*> test_argv_for_stack_trace;
  test_args_ref.reserve(test_file.test_args.size());
  test_argv_for_stack_trace.reserve(test_file.test_args.size() + 1);
  for (const auto& arg : test_file.test_args) {
    test_args_ref.push_back(arg);
    test_argv_for_stack_trace.push_back(arg.c_str());
  }
  // Add a trailing null so that this is a proper argv.
  test_argv_for_stack_trace.push_back(nullptr);

  // Add a stack trace entry for the test invocation.
  llvm::PrettyStackTraceProgram stack_trace_entry(
      test_argv_for_stack_trace.size() - 1, test_argv_for_stack_trace.data());

  // Conditionally capture console output. We use a scope exit to ensure the
  // captures terminate even on run failures.
  if (test_file.capture_console_output) {
    CaptureStderr();
    CaptureStdout();
  }

  // Prepare string streams to capture output. In order to address casting
  // constraints, we split calls to Run as a ternary based on whether we want to
  // capture output.
  llvm::raw_svector_ostream output_stream(test_file.actual_stdout);
  llvm::raw_svector_ostream error_stream(test_file.actual_stderr);

  ErrorOr<FileTestBase::RunResult> run_result =
      dump_output ? test_base.Run(test_args_ref, fs, input_stream, llvm::outs(),
                                  llvm::errs())
                  : test_base.Run(test_args_ref, fs, input_stream,
                                  output_stream, error_stream);

  // Ensure stdout/stderr are always fetched, even when discarded on error.
  if (test_file.capture_console_output) {
    // No need to flush stderr.
    llvm::outs().flush();
    test_file.actual_stdout += GetCapturedStdout();
    test_file.actual_stderr += GetCapturedStderr();
  }

  if (!run_result.ok()) {
    return std::move(run_result).error();
  }
  test_file.run_result = std::move(*run_result);
  return Success();
}

}  // namespace Carbon::Testing
