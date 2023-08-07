// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TESTING_FILE_TEST_AUTOUPDATE_H_
#define CARBON_TESTING_FILE_TEST_AUTOUPDATE_H_

#include <filesystem>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "testing/file_test/line.h"

namespace Carbon::Testing {

struct FileTestLineNumberReplacement {
  bool has_file;

  // The line replacement. The pattern should match lines. If has_file, pattern
  // should have a file and line group; otherwise, only a line group.
  std::string pattern;

  // line_formatv should provide {0} to substitute with [[@LINE...]] deltas.
  std::string line_formatv;
};

// Automatically updates CHECKs in the provided file. Returns true if updated.
auto AutoupdateFileTest(
    const std::filesystem::path& file_test_path, llvm::StringRef input_content,
    const llvm::SmallVector<llvm::StringRef>& filenames,
    int autoupdate_stdout_line_number, int autoupdate_stderr_line_number,
    llvm::SmallVector<llvm::SmallVector<FileTestLine>>& non_check_lines,
    llvm::StringRef stdout, llvm::StringRef stderr,
    FileTestLineNumberReplacement line_number_replacement,
    std::function<void(std::string&)> do_extra_check_replacements) -> bool;

}  // namespace Carbon::Testing

#endif  // CARBON_TESTING_FILE_TEST_AUTOUPDATE_H_
