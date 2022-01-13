//=- CFLAliasAnalysisUtils.h - Utilities for CFL Alias Analysis ----*- C++-*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// \file
// These are the utilities/helpers used by the CFL Alias Analyses available in
// tree, i.e. Steensgaard's and Andersens'.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CFLALIASANALYSISUTILS_H
#define LLVM_ANALYSIS_CFLALIASANALYSISUTILS_H

#include "llvm/IR/Function.h"
#include "llvm/IR/ValueHandle.h"

namespace llvm {
namespace cflaa {

template <typename AAResult> struct FunctionHandle final : public CallbackVH {
  FunctionHandle(Function *Fn, AAResult *Result)
      : CallbackVH(Fn), Result(Result) {
    assert(Fn != nullptr);
    assert(Result != nullptr);
  }

  void deleted() override { removeSelfFromCache(); }
  void allUsesReplacedWith(Value *) override { removeSelfFromCache(); }

private:
  AAResult *Result;

  void removeSelfFromCache() {
    assert(Result != nullptr);
    auto *Val = getValPtr();
    Result->evict(cast<Function>(Val));
    setValPtr(nullptr);
  }
};

static inline const Function *parentFunctionOfValue(const Value *Val) {
  if (auto *Inst = dyn_cast<Instruction>(Val)) {
    auto *Bb = Inst->getParent();
    return Bb->getParent();
  }

  if (auto *Arg = dyn_cast<Argument>(Val))
    return Arg->getParent();
  return nullptr;
} // namespace cflaa
} // namespace llvm
}

#endif // LLVM_ANALYSIS_CFLALIASANALYSISUTILS_H
