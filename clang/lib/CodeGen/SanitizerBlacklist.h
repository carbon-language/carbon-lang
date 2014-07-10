//===--- SanitizerBlacklist.h - Blacklist for sanitizers --------*- C++ -*-===//
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
#ifndef CLANG_CODEGEN_SANITIZERBLACKLIST_H
#define CLANG_CODEGEN_SANITIZERBLACKLIST_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SpecialCaseList.h"
#include <memory>

namespace llvm {
class GlobalVariable;
class Function;
class Module;
}

namespace clang {
namespace CodeGen {

class SanitizerBlacklist {
  std::unique_ptr<llvm::SpecialCaseList> SCL;

public:
  SanitizerBlacklist(llvm::SpecialCaseList *SCL) : SCL(SCL) {}
  bool isIn(const llvm::Module &M,
            const StringRef Category = StringRef()) const;
  bool isIn(const llvm::Function &F) const;
  bool isIn(const llvm::GlobalVariable &G,
            const StringRef Category = StringRef()) const;
  bool isBlacklistedType(StringRef MangledTypeName) const;
};
}  // end namespace CodeGen
}  // end namespace clang

#endif
