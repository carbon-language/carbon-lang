//===--- ARCMTActions.h - ARC Migrate Tool Frontend Actions -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ARCMIGRATE_ARCMT_ACTION_H
#define LLVM_CLANG_ARCMIGRATE_ARCMT_ACTION_H

#include "clang/Frontend/FrontendAction.h"
#include "llvm/ADT/OwningPtr.h"

namespace clang {
namespace arcmt {

class CheckAction : public WrapperFrontendAction {
protected:
  virtual bool BeginInvocation(CompilerInstance &CI);

public:
  CheckAction(FrontendAction *WrappedAction);
};

class ModifyAction : public WrapperFrontendAction {
protected:
  virtual bool BeginInvocation(CompilerInstance &CI);

public:
  ModifyAction(FrontendAction *WrappedAction);
};

class MigrateAction : public WrapperFrontendAction {
  std::string MigrateDir;
  std::string PlistOut;
  bool EmitPremigrationARCErros;
protected:
  virtual bool BeginInvocation(CompilerInstance &CI);

public:
  MigrateAction(FrontendAction *WrappedAction, StringRef migrateDir,
                StringRef plistOut,
                bool emitPremigrationARCErrors);
};

}
}

#endif
