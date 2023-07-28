// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TESTING_FILE_TEST_AUTOUPDATE_H_
#define CARBON_TESTING_FILE_TEST_AUTOUPDATE_H_

#include <filesystem>

#include "common/ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon::Testing {

// Interface for lines.
class FileTestLineBase {
 public:
  explicit FileTestLineBase(int line_number) : line_number_(line_number) {}
  virtual ~FileTestLineBase() {}

  // Prints the autoupdated line.
  virtual auto Print(llvm::raw_ostream& out) const -> void = 0;

  auto line_number() const -> int { return line_number_; }

 private:
  int line_number_;
};

// A line in the original file test.
class FileTestLine : public FileTestLineBase {
 public:
  explicit FileTestLine(int line_number, llvm::StringRef line)
      : FileTestLineBase(line_number), line_(line) {}

  auto Print(llvm::raw_ostream& out) const -> void override { out << line_; }

  auto indent() const -> llvm::StringRef {
    return line_.substr(0, line_.find_first_not_of(" \n"));
  }

 private:
  llvm::StringRef line_;
};

struct FileTestLineNumberReplacement {
  bool has_file;

  // The line replacement. The pattern should match lines. If has_file, pattern
  // should have a file and line group; otherwise, only a line group.
  std::string pattern;

  // sub_for_formatv should provide {0} to substitute with [[@LINE...]] deltas.
  std::string sub_for_formatv;
};

// Automatically updates CHECKs in the provided file. Returns true if updated.
auto AutoupdateFileTest(
    const std::filesystem::path& file_test_path, llvm::StringRef input_content,
    const llvm::SmallVector<llvm::StringRef>& filenames,
    int autoupdate_line_number,
    llvm::SmallVector<llvm::SmallVector<FileTestLine>>& non_check_lines,
    llvm::StringRef stdout, llvm::StringRef stderr,
    FileTestLineNumberReplacement line_number_replacement,
    std::function<void(std::string&)> do_extra_check_replacements) -> bool;

}  // namespace Carbon::Testing

#endif  // CARBON_TESTING_FILE_TEST_AUTOUPDATE_H_
