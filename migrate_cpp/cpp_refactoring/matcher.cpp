// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "migrate_cpp/cpp_refactoring/matcher.h"

#include "clang/Basic/SourceManager.h"

namespace Carbon {

void Matcher::AddReplacement(clang::CharSourceRange range,
                             llvm::StringRef replacement_text) {
  if (!range.isValid()) {
    // Invalid range.
    return;
  }
  if (GetSources().getDecomposedLoc(range.getBegin()).first !=
      GetSources().getDecomposedLoc(range.getEnd()).first) {
    // Range spans macro expansions.
    return;
  }
  if (GetSources().getFileID(range.getBegin()) !=
      GetSources().getFileID(range.getEnd())) {
    // Range spans files.
    return;
  }

  auto rep = clang::tooling::Replacement(
      GetSources(), GetSources().getExpansionRange(range), replacement_text);
  auto entry = replacements->find(std::string(rep.getFilePath()));
  if (entry == replacements->end()) {
    // The replacement was in a file which isn't being updated, such as a system
    // header.
    return;
  }

  llvm::Error err = entry->second.add(rep);
  if (err) {
    llvm::errs() << "Error with replacement `" << rep.toString()
                 << "`: " << llvm::toString(std::move(err)) << "\n";
  }
}

}  // namespace Carbon
