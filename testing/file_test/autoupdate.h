// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TESTING_FILE_TEST_AUTOUPDATE_H_
#define CARBON_TESTING_FILE_TEST_AUTOUPDATE_H_

#include <filesystem>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "re2/re2.h"
#include "testing/file_test/line.h"

namespace Carbon::Testing {

struct FileTestLineNumberReplacement {
  bool has_file;

  // The line replacement. The pattern should match lines. If has_file, pattern
  // should have a file and line group; otherwise, only a line group, but
  // default_file_re should be provided.
  //
  // Uses shared_ptr for storage in SmallVector.
  std::shared_ptr<RE2> re;

  // line_formatv should provide {0} to substitute with [[@LINE...]] deltas.
  std::string line_formatv;
};

// Automatically updates CHECKs in the provided file. Returns true if updated.
auto AutoupdateFileTest(
    const std::filesystem::path& file_test_path, llvm::StringRef input_content,
    const llvm::SmallVector<llvm::StringRef>& filenames,
    int autoupdate_line_number,
    const llvm::SmallVector<llvm::SmallVector<FileTestLine>>& non_check_lines,
    llvm::StringRef stdout, llvm::StringRef stderr,
    const std::optional<RE2>& default_file_re,
    const llvm::SmallVector<FileTestLineNumberReplacement>&
        line_number_replacements,
    std::function<void(std::string&)> do_extra_check_replacements) -> bool;

}  // namespace Carbon::Testing

#endif  // CARBON_TESTING_FILE_TEST_AUTOUPDATE_H_
