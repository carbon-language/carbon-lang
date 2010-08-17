//===- PCHDeserializationListener.h - Decl/Type PCH Read Events -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PCHDeserializationListener class, which is notified
//  by the PCHReader whenever a type or declaration is deserialized.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_PCH_DESERIALIZATION_LISTENER_H
#define LLVM_CLANG_FRONTEND_PCH_DESERIALIZATION_LISTENER_H

#include "clang/Serialization/PCHBitCodes.h"

namespace clang {

class Decl;
class PCHReader;
class QualType;

class PCHDeserializationListener {
protected:
  virtual ~PCHDeserializationListener() {}

public:
  /// \brief Tell the listener about the reader.
  virtual void SetReader(PCHReader *Reader) = 0;

  /// \brief An identifier was deserialized from the PCH.
  virtual void IdentifierRead(pch::IdentID ID, IdentifierInfo *II) = 0;
  /// \brief A type was deserialized from the PCH. The ID here has the qualifier
  ///        bits already removed, and T is guaranteed to be locally unqualified
  virtual void TypeRead(pch::TypeID ID, QualType T) = 0;
  /// \brief A decl was deserialized from the PCH.
  virtual void DeclRead(pch::DeclID ID, const Decl *D) = 0;
  /// \brief A selector was read from the PCH.
  virtual void SelectorRead(pch::SelectorID iD, Selector Sel) = 0;
};

}

#endif
