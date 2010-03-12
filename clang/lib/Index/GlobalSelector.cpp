//===-- GlobalSelector.cpp - Cross-translation-unit "token" for selectors -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  GlobalSelector is a ASTContext-independent way to refer to selectors.
//
//===----------------------------------------------------------------------===//

#include "clang/Index/GlobalSelector.h"
#include "ProgramImpl.h"
#include "clang/Index/Program.h"
#include "clang/AST/ASTContext.h"
using namespace clang;
using namespace idx;

/// \brief Get the ASTContext-specific selector.
Selector GlobalSelector::getSelector(ASTContext &AST) const {
  if (isInvalid())
    return Selector();

  Selector GlobSel = Selector(reinterpret_cast<uintptr_t>(Val));

  llvm::SmallVector<IdentifierInfo *, 8> Ids;
  for (unsigned i = 0, e = GlobSel.isUnarySelector() ? 1 : GlobSel.getNumArgs();
         i != e; ++i) {
    IdentifierInfo *GlobII = GlobSel.getIdentifierInfoForSlot(i);
    IdentifierInfo *II = &AST.Idents.get(GlobII->getName());
    Ids.push_back(II);
  }

  return AST.Selectors.getSelector(GlobSel.getNumArgs(), Ids.data());
}

/// \brief Get a printable name for debugging purpose.
std::string GlobalSelector::getPrintableName() const {
  if (isInvalid())
    return "<< Invalid >>";

  Selector GlobSel = Selector(reinterpret_cast<uintptr_t>(Val));
  return GlobSel.getAsString();
}

/// \brief Get a GlobalSelector for the ASTContext-specific selector.
GlobalSelector GlobalSelector::get(Selector Sel, Program &Prog) {
  if (Sel.isNull())
    return GlobalSelector();

  ProgramImpl &ProgImpl = *static_cast<ProgramImpl*>(Prog.Impl);

  llvm::SmallVector<IdentifierInfo *, 8> Ids;
  for (unsigned i = 0, e = Sel.isUnarySelector() ? 1 : Sel.getNumArgs();
         i != e; ++i) {
    IdentifierInfo *II = Sel.getIdentifierInfoForSlot(i);
    IdentifierInfo *GlobII = &ProgImpl.getIdents().get(II->getName());
    Ids.push_back(GlobII);
  }

  Selector GlobSel = ProgImpl.getSelectors().getSelector(Sel.getNumArgs(),
                                                         Ids.data());
  return GlobalSelector(GlobSel.getAsOpaquePtr());
}

unsigned
llvm::DenseMapInfo<GlobalSelector>::getHashValue(GlobalSelector Sel) {
  return DenseMapInfo<void*>::getHashValue(Sel.getAsOpaquePtr());
}
