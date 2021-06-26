// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "migrate_cpp/cpp_refactoring/matcher.h"

namespace ct = ::clang::tooling;

namespace Carbon {

void Matcher::AddReplacement(const clang::SourceManager& sm,
                             clang::CharSourceRange range,
                             llvm::StringRef replacement_text) {
  if (!range.isValid()) {
    // Invalid range.
    return;
  }
  if (sm.getDecomposedLoc(range.getBegin()).first !=
      sm.getDecomposedLoc(range.getEnd()).first) {
    // Range spans macro expansions.
    return;
  }
  if (sm.getFileID(range.getBegin()) != sm.getFileID(range.getEnd())) {
    // Range spans files.
    return;
  }

  auto rep = ct::Replacement(sm, sm.getExpansionRange(range), replacement_text);
  auto entry = replacements->find(std::string(rep.getFilePath()));
  if (entry == replacements->end()) {
    // The replacement was in a file which isn't being updated, such as a system
    // header.
    return;
  }

  auto err = entry->second.add(rep);
  if (err) {
    llvm::errs() << "Error with replacement `" << rep.toString()
                 << "`: " << llvm::toString(std::move(err)) << "\n";
  }
}

}  // namespace Carbon
