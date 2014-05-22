//===--- ClangTidyOptions.h - clang-tidy ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANG_TIDY_OPTIONS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANG_TIDY_OPTIONS_H

#include "llvm/Support/system_error.h"
#include <string>
#include <utility>
#include <vector>

namespace clang {
namespace tidy {

struct FileFilter {
  std::string Name;
  // LineRange is a pair<start, end> (inclusive).
  typedef std::pair<unsigned, unsigned> LineRange;
  std::vector<LineRange> LineRanges;
};

/// \brief Contains options for clang-tidy.
struct ClangTidyOptions {
  ClangTidyOptions() : Checks("*"), AnalyzeTemporaryDtors(false) {}
  std::string Checks;

  // Output warnings from headers matching this filter. Warnings from main files
  // will always be displayed.
  std::string HeaderFilterRegex;

  // Output warnings from certain line ranges of certain files only. If this
  // list is emtpy, it won't be applied.
  std::vector<FileFilter> LineFilter;

  bool AnalyzeTemporaryDtors;
};

/// \brief Parses LineFilter from JSON and stores it to the \c Options.
llvm::error_code parseLineFilter(const std::string &LineFilter,
                                 clang::tidy::ClangTidyOptions &Options);

} // end namespace tidy
} // end namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANG_TIDY_OPTIONS_H
