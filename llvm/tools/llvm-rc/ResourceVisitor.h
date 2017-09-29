//===-- ResourceVisitor.h ---------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
//
// This defines a base class visiting resource script resources.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMRC_RESOURCEVISITOR_H
#define LLVM_TOOLS_LLVMRC_RESOURCEVISITOR_H

#include "llvm/Support/Error.h"

namespace llvm {
namespace rc {

class RCResource;

class LanguageResource;

class Visitor {
public:
  virtual Error visitNullResource(const RCResource *) = 0;
  virtual Error visitHTMLResource(const RCResource *) = 0;

  virtual Error visitLanguageStmt(const LanguageResource *) = 0;

  virtual ~Visitor() {}
};

} // namespace rc
} // namespace llvm

#endif
