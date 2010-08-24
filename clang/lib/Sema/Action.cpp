//===--- Action.cpp - Implement the Action class --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Action interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/Action.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Scope.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/RecyclingAllocator.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

void PrettyStackTraceActionsDecl::print(llvm::raw_ostream &OS) const {
  if (Loc.isValid()) {
    Loc.print(OS, SM);
    OS << ": ";
  }
  OS << Message;

  std::string Name = Actions.getDeclName(TheDecl);
  if (!Name.empty())
    OS << " '" << Name << '\'';

  OS << '\n';
}

///  Out-of-line virtual destructor to provide home for Action class.
Action::~Action() {}
