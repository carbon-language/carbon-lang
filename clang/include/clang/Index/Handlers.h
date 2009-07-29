//===--- Handlers.h - Interfaces for receiving information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Abstract interfaces for receiving information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_HANDLERS_H
#define LLVM_CLANG_INDEX_HANDLERS_H

namespace clang {

namespace idx {
  class Entity;
  class TranslationUnit;

/// \brief Abstract interface for receiving Entities.
class EntityHandler {
public:
  virtual ~EntityHandler();
  virtual void Handle(Entity Ent) = 0;
};

/// \brief Abstract interface for receiving TranslationUnits.
class TranslationUnitHandler {
public:
  virtual ~TranslationUnitHandler();
  virtual void Handle(TranslationUnit *TU) = 0;
};
  
} // namespace idx

} // namespace clang

#endif
