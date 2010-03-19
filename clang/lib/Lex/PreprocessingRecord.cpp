//===--- PreprocessingRecord.cpp - Record of Preprocessing ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the PreprocessingRecord class, which maintains a record
//  of what occurred during preprocessing, and its helpers.
//
//===----------------------------------------------------------------------===//
#include "clang/Lex/PreprocessingRecord.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Token.h"

using namespace clang;

void PreprocessingRecord::addPreprocessedEntity(PreprocessedEntity *Entity) {
  PreprocessedEntities.push_back(Entity);
}

MacroDefinition *PreprocessingRecord::findMacroDefinition(MacroInfo *MI) {
  llvm::DenseMap<const MacroInfo *, MacroDefinition *>::iterator Pos
    = MacroDefinitions.find(MI);
  if (Pos == MacroDefinitions.end())
    return 0;
  
  return Pos->second;
}

void PreprocessingRecord::MacroExpands(const Token &Id, const MacroInfo* MI) {
  PreprocessedEntities.push_back(
                       new (*this) MacroInstantiation(Id.getIdentifierInfo(),
                                                      Id.getLocation(),
                                                      MacroDefinitions[MI]));
}

void PreprocessingRecord::MacroDefined(const IdentifierInfo *II, 
                                       const MacroInfo *MI) {
  SourceRange R(MI->getDefinitionLoc(), MI->getDefinitionEndLoc());
  MacroDefinition *Def
    = new (*this) MacroDefinition(II, MI->getDefinitionLoc(), R);
  MacroDefinitions[MI] = Def;
  PreprocessedEntities.push_back(Def);
}
