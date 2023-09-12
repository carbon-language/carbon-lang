// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_DRIVER_FILE_TEST_BASE_H_
#define CARBON_TOOLCHAIN_DRIVER_DRIVER_FILE_TEST_BASE_H_

#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "testing/file_test/file_test_base.h"
#include "toolchain/driver/driver.h"

namespace Carbon::Testing {
namespace {

// Provides common test support for the driver. This is used by file tests in
// phase subdirectories.
class ToolchainFileTest : public FileTestBase {
 public:
  explicit ToolchainFileTest(llvm::StringRef test_name)
      : FileTestBase(test_name), component_(GetComponent(test_name)) {}

  auto Run(const llvm::SmallVector<llvm::StringRef>& test_args,
           llvm::vfs::InMemoryFileSystem& fs, llvm::raw_pwrite_stream& stdout,
           llvm::raw_pwrite_stream& stderr) -> ErrorOr<bool> override {
    Driver driver(fs, stdout, stderr);
    return driver.RunCommand(test_args);
  }

  auto GetDefaultArgs() -> llvm::SmallVector<std::string> override {
    if (component_ == "check") {
      return {"compile", "--phase=check", "--dump-sem-ir", "%s"};
    } else if (component_ == "lex") {
      return {"compile", "--phase=lex", "--dump-tokens", "%s"};
    } else if (component_ == "lower") {
      return {"compile", "--phase=lower", "--dump-llvm-ir", "%s"};
    } else if (component_ == "parse") {
      return {"compile", "--phase=parse", "--dump-parse-tree", "%s"};
    } else if (component_ == "codegen" || component_ == "driver") {
      CARBON_FATAL() << "ARGS is always set in these tests";
    } else {
      CARBON_FATAL() << "Unexpected test component " << component_ << ": "
                     << test_name();
    }
  }

  auto GetDefaultFileRE(llvm::ArrayRef<llvm::StringRef> filenames)
      -> llvm::SmallVector<std::shared_ptr<RE2>> override {
    llvm::SmallVector<std::shared_ptr<RE2>> res;
    if (component_ == "lex" || component_ == "parse" || component_ == "check") {
      res.push_back(std::make_shared<RE2>(
          llvm::formatv(R"(^- filename: ({0})$)", llvm::join(filenames, "|"))));
    }
    if (component_ == "check") {
      // SemIR formats files differently for textual versus raw.
      res.push_back(std::make_shared<RE2>(llvm::formatv(
          R"re(^file "({0})" {$)re", llvm::join(filenames, "|"))));
    }
    return res;
  }

  auto GetLineNumberReplacements(llvm::ArrayRef<llvm::StringRef> filenames)
      -> llvm::SmallVector<LineNumberReplacement> override {
    auto replacements = FileTestBase::GetLineNumberReplacements(filenames);
    if (component_ == "lex") {
      replacements.push_back({.has_file = false,
                              .re = std::make_shared<RE2>(R"(line: (\s*\d+))"),
                              // The `{{{{` becomes `{{`.
                              .line_formatv = "{{{{ *}}{0}"});
    } else if (component_ == "check") {
      replacements.push_back({.has_file = false,
                              .re = std::make_shared<RE2>(R"(\Wloc(\d+))"),
                              .line_formatv = "{0}"});
    }
    return replacements;
  }

  auto DoExtraCheckReplacements(std::string& check_line) -> void override {
    if (component_ == "driver") {
      // TODO: Disable token output, it's not interesting for these tests.
      if (llvm::StringRef(check_line).starts_with("// CHECK:STDOUT: {")) {
        check_line = "// CHECK:STDOUT: {{.*}}";
      }
    } else if (component_ == "lex") {
      // Ignore the resulting column of EndOfFile because it's often the end of
      // the CHECK comment.
      static RE2 end_of_file_re(R"((EndOfFile.*column: )( *\d+))");
      RE2::Replace(&check_line, end_of_file_re, R"(\1{{ *\\d+}})");
    } else {
      return FileTestBase::DoExtraCheckReplacements(check_line);
    }
  }

 private:
  // Returns the toolchain subdirectory being tested.
  static auto GetComponent(llvm::StringRef test_name) -> llvm::StringRef {
    CARBON_CHECK(test_name.consume_front("toolchain/"));
    test_name = test_name.take_front(test_name.find("/"));
    return test_name;
  }

  const llvm::StringRef component_;
};

}  // namespace

CARBON_FILE_TEST_FACTORY(ToolchainFileTest);

}  // namespace Carbon::Testing

#endif  // CARBON_TOOLCHAIN_DRIVER_DRIVER_FILE_TEST_BASE_H_
