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

#include "llvm/ADT/DenseSet.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <functional>

namespace llvm {

class Function;
class GlobalVariable;
class Module;

namespace orc {

/// @brief Functor type for describing how CloneSubModule should mutate a
///        GlobalVariable.
typedef std::function<void(GlobalVariable &, const GlobalVariable &,
                           ValueToValueMapTy &)> HandleGlobalVariableFtor;

/// @brief Functor type for describing how CloneSubModule should mutate a
///        Function.
typedef std::function<void(Function &, const Function &, ValueToValueMapTy &)>
    HandleFunctionFtor;

/// @brief Copies the initializer from Orig to New.
///
///   Type is suitable for implicit conversion to a HandleGlobalVariableFtor.
void copyGVInitializer(GlobalVariable &New, const GlobalVariable &Orig,
                       ValueToValueMapTy &VMap);

/// @brief Copies the body of Orig to New.
///
///   Type is suitable for implicit conversion to a HandleFunctionFtor.
void copyFunctionBody(Function &New, const Function &Orig,
                      ValueToValueMapTy &VMap);

/// @brief Clone a subset of the module Src into Dst.
void CloneSubModule(Module &Dst, const Module &Src,
                    HandleGlobalVariableFtor HandleGlobalVariable,
                    HandleFunctionFtor HandleFunction, bool KeepInlineAsm);

} // End namespace orc.
} // End namespace llvm.

#endif // LLVM_EXECUTIONENGINE_ORC_CLONESUBMODULE_H
