//===------ OrcTestCommon.h - Utilities for Orc Unit Tests ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Common utilities for the Orc unit tests.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_UNITTESTS_EXECUTIONENGINE_ORC_ORCTESTCOMMON_H
#define LLVM_UNITTESTS_EXECUTIONENGINE_ORC_ORCTESTCOMMON_H

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/TypeBuilder.h"
#include <memory>

namespace llvm {

  class ModuleBuilder {
  public:
    ModuleBuilder(LLVMContext &Context, StringRef Triple,
                  StringRef Name);

    template <typename FuncType>
    Function* createFunctionDecl(Module *M, StringRef Name) {
      return Function::Create(
               TypeBuilder<FuncType, false>::get(M->getContext()),
               GlobalValue::ExternalLinkage, Name, M);
    }

    Module* getModule() { return M.get(); }
    const Module* getModule() const { return M.get(); }
    std::unique_ptr<Module> takeModule() { return std::move(M); }

  private:
    std::unique_ptr<Module> M;
    IRBuilder<> Builder;
  };

  // Dummy struct type.
  struct DummyStruct {
    int X[256];
  };

  // TypeBuilder specialization for DummyStruct.
  template <bool XCompile>
  class TypeBuilder<DummyStruct, XCompile> {
  public:
    static StructType *get(LLVMContext &Context) {
      return StructType::get(
          TypeBuilder<types::i<32>[256], XCompile>::get(Context), nullptr);
    }
  };


} // namespace llvm

#endif
