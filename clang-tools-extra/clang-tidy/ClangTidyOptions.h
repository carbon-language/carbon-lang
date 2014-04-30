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

namespace clang {
namespace tidy {

/// \brief Contains options for clang-tidy.
struct ClangTidyOptions {
  ClangTidyOptions() : EnableChecksRegex(".*"), AnalyzeTemporaryDtors(false) {}
  std::string EnableChecksRegex;
  std::string DisableChecksRegex;
  bool AnalyzeTemporaryDtors;
};

} // end namespace tidy
} // end namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANG_TIDY_OPTIONS_H
