//===-- ReplaceAutoPtr.cpp ---------- std::auto_ptr replacement -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the implementation of the ReplaceAutoPtrTransform
/// class.
///
//===----------------------------------------------------------------------===//

#include "ReplaceAutoPtr.h"
#include "ReplaceAutoPtrActions.h"
#include "ReplaceAutoPtrMatchers.h"

#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"

using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;

int
ReplaceAutoPtrTransform::apply(FileOverrides &InputStates,
                               const CompilationDatabase &Database,
                               const std::vector<std::string> &SourcePaths) {
  ClangTool Tool(Database, SourcePaths);

  unsigned AcceptedChanges = 0;

  MatchFinder Finder;
  AutoPtrReplacer Replacer(getReplacements(), AcceptedChanges,
                           /*Owner=*/*this);
  OwnershipTransferFixer Fixer(getReplacements(), AcceptedChanges,
                                   /*Owner=*/*this);

  Finder.addMatcher(makeAutoPtrTypeLocMatcher(), &Replacer);
  Finder.addMatcher(makeAutoPtrUsingDeclMatcher(), &Replacer);
  Finder.addMatcher(makeTransferOwnershipExprMatcher(), &Fixer);

  setOverrides(InputStates);

  if (Tool.run(createActionFactory(Finder))) {
    llvm::errs() << "Error encountered during translation.\n";
    return 1;
  }

  setAcceptedChanges(AcceptedChanges);

  return 0;
}
