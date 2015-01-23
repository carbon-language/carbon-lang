//===-- CloneSubModule.h - Utilities for extracting sub-modules -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Contains utilities for extracting sub-modules. Useful for breaking up modules
// for lazy jitting.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_CLONESUBMODULE_H
#define LLVM_EXECUTIONENGINE_ORC_CLONESUBMODULE_H

#include "llvm/Transforms/Utils/ValueMapper.h"
#include <functional>

namespace llvm {

class Function;
class GlobalVariable;
class Module;

typedef std::function<void(GlobalVariable &, const GlobalVariable &,
                           ValueToValueMapTy &)> HandleGlobalVariableFtor;

typedef std::function<void(Function &, const Function &, ValueToValueMapTy &)>
    HandleFunctionFtor;

void copyGVInitializer(GlobalVariable &New, const GlobalVariable &Orig,
                       ValueToValueMapTy &VMap);

void copyFunctionBody(Function &New, const Function &Orig,
                      ValueToValueMapTy &VMap);

std::unique_ptr<Module>
CloneSubModule(const Module &M, HandleGlobalVariableFtor HandleGlobalVariable,
               HandleFunctionFtor HandleFunction, bool KeepInlineAsm);
}

#endif // LLVM_EXECUTIONENGINE_ORC_CLONESUBMODULE_H
