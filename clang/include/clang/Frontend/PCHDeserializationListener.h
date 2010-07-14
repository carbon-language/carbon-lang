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

#include "clang/Frontend/PCHBitCodes.h"

namespace clang {

class Decl;
class QualType;

class PCHDeserializationListener {
protected:
  ~PCHDeserializationListener() {}

public:
  virtual void TypeRead(pch::TypeID ID, QualType T) = 0;
  virtual void DeclRead(pch::DeclID ID, const Decl *D) = 0;
};

}

#endif
