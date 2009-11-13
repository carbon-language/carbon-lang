//===--- CompilerInstance.cpp ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/LLVMContext.h"
using namespace clang;

CompilerInstance::CompilerInstance(llvm::LLVMContext *_LLVMContext,
                                   bool _OwnsLLVMContext)
  : LLVMContext(_LLVMContext),
    OwnsLLVMContext(_OwnsLLVMContext) {
}

CompilerInstance::~CompilerInstance() {
  if (OwnsLLVMContext)
    delete LLVMContext;
}

void CompilerInstance::createFileManager() {
  FileMgr.reset(new FileManager());
}

void CompilerInstance::createSourceManager() {
  SourceMgr.reset(new SourceManager());
}
