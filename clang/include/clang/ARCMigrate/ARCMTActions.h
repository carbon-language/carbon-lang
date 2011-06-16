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
  virtual void ExecuteAction();

public:
  CheckAction(FrontendAction *WrappedAction);
};

class TransformationAction : public WrapperFrontendAction {
protected:
  virtual void ExecuteAction();

public:
  TransformationAction(FrontendAction *WrappedAction);
};

class InMemoryTransformationAction : public WrapperFrontendAction {
protected:
  virtual void ExecuteAction();

public:
  InMemoryTransformationAction(FrontendAction *WrappedAction);
};

}
}

#endif
