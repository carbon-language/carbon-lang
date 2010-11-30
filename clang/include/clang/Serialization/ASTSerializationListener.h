//===- ASTSerializationListener.h - Decl/Type PCH Write Events -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ASTSerializationListener class, which is notified
//  by the ASTWriter when an entity is serialized.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_FRONTEND_AST_SERIALIZATION_LISTENER_H
#define LLVM_CLANG_FRONTEND_AST_SERIALIZATION_LISTENER_H

#include "llvm/Support/DataTypes.h"

namespace clang {

class PreprocessedEntity;
  
/// \brief Listener object that receives callbacks when certain kinds of 
/// entities are serialized.
class ASTSerializationListener {
public:
  virtual ~ASTSerializationListener();
  
  /// \brief Callback invoked whenever a preprocessed entity is serialized.
  ///
  /// This callback will only occur when the translation unit was created with
  /// a detailed preprocessing record.
  ///
  /// \param Entity The entity that has been serialized.
  ///
  /// \param Offset The offset (in bits) of this entity in the resulting
  /// AST file.
  virtual void SerializedPreprocessedEntity(PreprocessedEntity *Entity,
                                            uint64_t Offset) = 0;
};

}

#endif
