//===--- Program.h - Entity originator and misc -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Storage for Entities and utility functions
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_PROGRAM_H
#define LLVM_CLANG_INDEX_PROGRAM_H

namespace clang {
  class ASTContext;

namespace idx {
  class EntityHandler;

/// \brief Repository for Entities.
class Program {
  void *Impl;

  Program(const Program&); // do not implement
  Program &operator=(const Program &); // do not implement
  friend class Entity;
  
public:
  Program();
  ~Program();

  /// \brief Traverses the AST and passes all the entities to the Handler.
  void FindEntities(ASTContext &Ctx, EntityHandler *Handler);
};

} // namespace idx

} // namespace clang

#endif
