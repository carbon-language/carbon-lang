//===--- tools/extra/clang-rename/RenamingAction.h - Clang rename tool ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Provides an action to rename every symbol at a point.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_RENAME_RENAMING_ACTION_H_
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_RENAME_RENAMING_ACTION_H_

#include "clang/Tooling/Refactoring.h"

namespace clang {
class ASTConsumer;
class CompilerInstance;

namespace rename {

class RenamingAction {
public:
  RenamingAction(const std::string &NewName, const std::string &PrevName,
                 const std::vector<std::string> &USRs,
                 tooling::Replacements &Replaces, bool PrintLocations = false)
      : NewName(NewName), PrevName(PrevName), USRs(USRs), Replaces(Replaces),
        PrintLocations(PrintLocations) {
  }

  std::unique_ptr<ASTConsumer> newASTConsumer();

private:
  const std::string &NewName, &PrevName;
  const std::vector<std::string> &USRs;
  tooling::Replacements &Replaces;
  bool PrintLocations;
};

}
}

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_RENAME_RENAMING_ACTION_H_
