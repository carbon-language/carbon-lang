//===--- Program.h - Cross-translation unit information ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the idx::Program interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_PROGRAM_H
#define LLVM_CLANG_INDEX_PROGRAM_H

namespace clang {
  class ASTContext;

namespace idx {
  class EntityHandler;

/// \brief Top level object that owns and maintains information
/// that is common across translation units.
class Program {
  void *Impl;

  Program(const Program&); // do not implement
  Program &operator=(const Program &); // do not implement
  friend class Entity;
  friend class GlobalSelector;

public:
  Program();
  ~Program();

  /// \brief Traverses the AST and passes all the entities to the Handler.
  void FindEntities(ASTContext &Ctx, EntityHandler &Handler);
};

} // namespace idx

} // namespace clang

#endif
