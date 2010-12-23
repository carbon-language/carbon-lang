//===-- AnalysisManager.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/EntoSA/PathSensitive/AnalysisManager.h"
#include "clang/Index/Entity.h"
#include "clang/Index/Indexer.h"

using namespace clang;
using namespace ento;

AnalysisContext *
AnalysisManager::getAnalysisContextInAnotherTU(const Decl *D) {
  idx::Entity Ent = idx::Entity::get(const_cast<Decl *>(D), 
                                     Idxer->getProgram());
  FunctionDecl *FuncDef;
  idx::TranslationUnit *TU;
  llvm::tie(FuncDef, TU) = Idxer->getDefinitionFor(Ent);

  if (FuncDef == 0)
    return 0;

  // This AnalysisContext wraps function definition in another translation unit.
  // But it is still owned by the AnalysisManager associated with the current
  // translation unit.
  return AnaCtxMgr.getContext(FuncDef, TU);
}
