//===--- ARCMTActions.cpp - ARC Migrate Tool Frontend Actions ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/ARCMigrate/ARCMTActions.h"
#include "clang/ARCMigrate/ARCMT.h"
#include "clang/Frontend/CompilerInstance.h"

using namespace clang;
using namespace arcmt;

void CheckAction::ExecuteAction() {
  CompilerInstance &CI = getCompilerInstance();
  if (arcmt::checkForManualIssues(CI.getInvocation(), getCurrentFile(),
                                  getCurrentFileKind(),
                                  CI.getDiagnostics().getClient()))
    return;

  // We only want to see warnings reported from arcmt::checkForManualIssues.
  CI.getDiagnostics().setIgnoreAllWarnings(true);
  WrapperFrontendAction::ExecuteAction();
}

CheckAction::CheckAction(FrontendAction *WrappedAction)
  : WrapperFrontendAction(WrappedAction) {}

void TransformationAction::ExecuteAction() {
  CompilerInstance &CI = getCompilerInstance();
  if (arcmt::applyTransformations(CI.getInvocation(), getCurrentFile(),
                                  getCurrentFileKind(),
                                  CI.getDiagnostics().getClient()))
    return;

  WrapperFrontendAction::ExecuteAction();
}

TransformationAction::TransformationAction(FrontendAction *WrappedAction)
  : WrapperFrontendAction(WrappedAction) {}
