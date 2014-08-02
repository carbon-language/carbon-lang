//===--- SanitizerMetadata.cpp - Blacklist for sanitizers -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Class which emits metadata consumed by sanitizer instrumentation passes.
//
//===----------------------------------------------------------------------===//
#include "SanitizerMetadata.h"
#include "CodeGenModule.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"

using namespace clang;
using namespace CodeGen;

SanitizerMetadata::SanitizerMetadata(CodeGenModule &CGM) : CGM(CGM) {}

void SanitizerMetadata::reportGlobalToASan(llvm::GlobalVariable *GV,
                                           SourceLocation Loc, StringRef Name,
                                           bool IsDynInit, bool IsBlacklisted) {
  if (!CGM.getLangOpts().Sanitize.Address)
    return;
  IsDynInit &= !CGM.getSanitizerBlacklist().isIn(*GV, "init");
  IsBlacklisted |= CGM.getSanitizerBlacklist().isIn(*GV);

  llvm::Value *LocDescr = nullptr;
  llvm::Value *GlobalName = nullptr;
  llvm::LLVMContext &VMContext = CGM.getLLVMContext();
  if (!IsBlacklisted) {
    // Don't generate source location and global name if it is blacklisted -
    // it won't be instrumented anyway.
    LocDescr = getLocationMetadata(Loc);
    if (!Name.empty())
      GlobalName = llvm::MDString::get(VMContext, Name);
  }

  llvm::Value *GlobalMetadata[] = {
      GV, LocDescr, GlobalName,
      llvm::ConstantInt::get(llvm::Type::getInt1Ty(VMContext), IsDynInit),
      llvm::ConstantInt::get(llvm::Type::getInt1Ty(VMContext), IsBlacklisted)};

  llvm::MDNode *ThisGlobal = llvm::MDNode::get(VMContext, GlobalMetadata);
  llvm::NamedMDNode *AsanGlobals =
      CGM.getModule().getOrInsertNamedMetadata("llvm.asan.globals");
  AsanGlobals->addOperand(ThisGlobal);
}

void SanitizerMetadata::reportGlobalToASan(llvm::GlobalVariable *GV,
                                           const VarDecl &D, bool IsDynInit) {
  if (!CGM.getLangOpts().Sanitize.Address)
    return;
  std::string QualName;
  llvm::raw_string_ostream OS(QualName);
  D.printQualifiedName(OS);
  reportGlobalToASan(GV, D.getLocation(), OS.str(), IsDynInit);
}

void SanitizerMetadata::disableSanitizerForGlobal(llvm::GlobalVariable *GV) {
  // For now, just make sure the global is not modified by the ASan
  // instrumentation.
  if (CGM.getLangOpts().Sanitize.Address)
    reportGlobalToASan(GV, SourceLocation(), "", false, true);
}

llvm::MDNode *SanitizerMetadata::getLocationMetadata(SourceLocation Loc) {
  PresumedLoc PLoc = CGM.getContext().getSourceManager().getPresumedLoc(Loc);
  if (!PLoc.isValid())
    return nullptr;
  llvm::LLVMContext &VMContext = CGM.getLLVMContext();
  llvm::Value *LocMetadata[] = {
      llvm::MDString::get(VMContext, PLoc.getFilename()),
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), PLoc.getLine()),
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext),
                             PLoc.getColumn()),
  };
  return llvm::MDNode::get(VMContext, LocMetadata);
}
