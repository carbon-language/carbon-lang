//===--- SanitizerBlacklist.cpp - Blacklist for sanitizers ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// User-provided blacklist used to disable/alter instrumentation done in
// sanitizers.
//
//===----------------------------------------------------------------------===//
#include "SanitizerBlacklist.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"

using namespace clang;
using namespace CodeGen;

static StringRef GetGlobalTypeString(const llvm::GlobalValue &G) {
  // Types of GlobalVariables are always pointer types.
  llvm::Type *GType = G.getType()->getElementType();
  // For now we support blacklisting struct types only.
  if (llvm::StructType *SGType = dyn_cast<llvm::StructType>(GType)) {
    if (!SGType->isLiteral())
      return SGType->getName();
  }
  return "<unknown type>";
}

bool SanitizerBlacklist::isIn(const llvm::Module &M,
                              const StringRef Category) const {
  return SCL->inSection("src", M.getModuleIdentifier(), Category);
}

bool SanitizerBlacklist::isIn(const llvm::Function &F) const {
  return isIn(*F.getParent()) ||
         SCL->inSection("fun", F.getName(), "");
}

bool SanitizerBlacklist::isIn(const llvm::GlobalVariable &G,
                              const StringRef Category) const {
  return isIn(*G.getParent(), Category) ||
         SCL->inSection("global", G.getName(), Category) ||
         SCL->inSection("type", GetGlobalTypeString(G), Category);
}

bool SanitizerBlacklist::isBlacklistedType(StringRef MangledTypeName) const {
  return SCL->inSection("type", MangledTypeName);
}
