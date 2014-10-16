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
#include "clang/Basic/SanitizerBlacklist.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"

using namespace clang;

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

SanitizerBlacklist::SanitizerBlacklist(const std::string &BlacklistPath)
    : SCL(llvm::SpecialCaseList::createOrDie(BlacklistPath)) {}

bool SanitizerBlacklist::isIn(const llvm::Function &F) const {
  return isBlacklistedFile(F.getParent()->getModuleIdentifier()) ||
         isBlacklistedFunction(F.getName());
}

bool SanitizerBlacklist::isIn(const llvm::GlobalVariable &G,
                              StringRef Category) const {
  return isBlacklistedFile(G.getParent()->getModuleIdentifier(), Category) ||
         SCL->inSection("global", G.getName(), Category) ||
         SCL->inSection("type", GetGlobalTypeString(G), Category);
}

bool SanitizerBlacklist::isBlacklistedType(StringRef MangledTypeName) const {
  return SCL->inSection("type", MangledTypeName);
}

bool SanitizerBlacklist::isBlacklistedFunction(StringRef FunctionName) const {
  return SCL->inSection("fun", FunctionName);
}

bool SanitizerBlacklist::isBlacklistedFile(StringRef FileName,
                                           StringRef Category) const {
  return SCL->inSection("src", FileName, Category);
}
