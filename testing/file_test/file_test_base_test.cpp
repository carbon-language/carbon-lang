// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/file_test/file_test_base.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/ostream.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"

namespace Carbon::Testing {
namespace {

class FileTestBaseTest : public FileTestBase {
 public:
  using FileTestBase::FileTestBase;

  auto Run(const llvm::SmallVector<llvm::StringRef>& test_args,
           llvm::vfs::InMemoryFileSystem& fs, llvm::raw_pwrite_stream& stdout,
           llvm::raw_pwrite_stream& stderr) -> ErrorOr<bool> override {
    llvm::ArrayRef<llvm::StringRef> args = test_args;

    llvm::ListSeparator sep;
    stdout << args.size() << " args: ";
    for (auto arg : args) {
      stdout << sep << "`" << arg << "`";
    }
    stdout << "\n";

    auto filename = std::filesystem::path(test_name().str()).filename();
    if (filename == "args.carbon") {
      // 'args.carbon' has custom arguments, so don't do regular argument
      // validation for it.
      return true;
    }

    if (args.empty() || args.front() != "default_args") {
      return ErrorBuilder() << "missing `default_args` argument";
    }
    args = args.drop_front();

    for (auto arg : args) {
      if (!fs.exists(arg)) {
        return ErrorBuilder() << "Missing file: " << arg;
      }
    }

    if (filename == "example.carbon") {
      int delta_line = 10;
      stdout << "something\n"
             << "\n"
             << "example.carbon:" << delta_line + 1 << ": Line delta\n"
             << "example.carbon:" << delta_line << ": Negative line delta\n"
             << "+*[]{}\n"
             << "Foo baz\n";
      return true;
    } else if (filename == "fail_example.carbon") {
      stderr << "Oops\n";
      return false;
    } else if (filename == "two_files.carbon" ||
               filename == "not_split.carbon") {
      for (auto arg : args) {
        // Describe file contents to stdout to validate splitting.
        auto file = fs.getBufferForFile(arg, /*FileSize=*/-1,
                                        /*RequiresNullTerminator=*/false);
        if (file.getError()) {
          return Error(file.getError().message());
        }
        llvm::StringRef content = file.get()->getBuffer();
        stdout << arg << ":1: starts with \"";
        stdout.write_escaped(content.take_front(40));
        stdout << "\", length " << content.count('\n') << " lines\n";
      }
      return true;
    } else if (filename == "alternating_files.carbon") {
      stdout << "unattached message 1\n"
             << "a.carbon:2: message 2\n"
             << "b.carbon:5: message 3\n"
             << "a.carbon:2: message 4\n"
             << "b.carbon:5: message 5\n"
             << "unattached message 6\n";
      stderr << "unattached message 1\n"
             << "a.carbon:2: message 2\n"
             << "b.carbon:5: message 3\n"
             << "a.carbon:2: message 4\n"
             << "b.carbon:5: message 5\n"
             << "unattached message 6\n";
      return true;
    } else if (filename == "unattached_multi_file.carbon") {
      stdout << "unattached message 1\n"
             << "unattached message 2\n";
      stderr << "unattached message 3\n"
             << "unattached message 4\n";
      return true;
    } else if (filename == "file_only_re_one_file.carbon") {
      stdout << "unattached message 1\n"
             << "file: file_only_re_one_file.carbon\n"
             << "line: 1\n"
             << "unattached message 2\n";
      return true;
    } else if (filename == "file_only_re_multi_file.carbon") {
      int msg_count = 0;
      stdout << "unattached message " << ++msg_count << "\n"
             << "file: a.carbon\n"
             << "unattached message " << ++msg_count << "\n"
             << "line: 3: attached message " << ++msg_count << "\n"
             << "unattached message " << ++msg_count << "\n"
             << "line: 8: late message " << ++msg_count << "\n"
             << "unattached message " << ++msg_count << "\n"
             << "file: b.carbon\n"
             << "line: 2: attached message " << ++msg_count << "\n"
             << "unattached message " << ++msg_count << "\n"
             << "line: 7: late message " << ++msg_count << "\n"
             << "unattached message " << ++msg_count << "\n";
      return true;
    } else {
      return ErrorBuilder() << "Unexpected file: " << filename;
    }
  }

  auto GetDefaultArgs() -> llvm::SmallVector<std::string> override {
    return {"default_args", "%s"};
  }

  auto GetDefaultFileRE(llvm::ArrayRef<llvm::StringRef> filenames)
      -> std::optional<RE2> override {
    return std::make_optional<RE2>(
        llvm::formatv(R"(file: ({0}))", llvm::join(filenames, "|")));
  }

  auto GetLineNumberReplacements(llvm::ArrayRef<llvm::StringRef> filenames)
      -> llvm::SmallVector<LineNumberReplacement> override {
    auto replacements = FileTestBase::GetLineNumberReplacements(filenames);
    auto filename = std::filesystem::path(test_name().str()).filename();
    if (llvm::StringRef(filename).starts_with("file_only_re_")) {
      replacements.push_back({.has_file = false,
                              .re = std::make_shared<RE2>(R"(line: (\d+))"),
                              .line_formatv = "{0}"});
    }
    return replacements;
  }
};

}  // namespace

CARBON_FILE_TEST_FACTORY(FileTestBaseTest);

}  // namespace Carbon::Testing
