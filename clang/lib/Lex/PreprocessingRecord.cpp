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

void PopulatePreprocessingRecord::MacroExpands(const Token &Id, 
                                               const MacroInfo* MI) {
  Record.addPreprocessedEntity(
                        new (Record) MacroInstantiation(Id.getIdentifierInfo(),
                                                       Id.getLocation(),
                                                      MI->getDefinitionLoc()));
}

void PopulatePreprocessingRecord::MacroDefined(const IdentifierInfo *II, 
                                               const MacroInfo *MI) {
  SourceRange R(MI->getDefinitionLoc(), MI->getDefinitionEndLoc());
  Record.addPreprocessedEntity(
                  new (Record) MacroDefinition(II, MI->getDefinitionLoc(), R));
}
