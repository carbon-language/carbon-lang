//===-------- OrcCBindingsStack.cpp - Orc JIT stack for C bindings --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OrcCBindingsStack.h"

#include "llvm/ExecutionEngine/Orc/OrcTargetSupport.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DynamicLibrary.h"
#include <cstdio>
#include <system_error>

using namespace llvm;

OrcCBindingsStack::CallbackManagerBuilder
OrcCBindingsStack::createCallbackManagerBuilder(Triple T) {
  switch (T.getArch()) {
    default: return nullptr;

    case Triple::x86_64: {
      typedef orc::JITCompileCallbackManager<CompileLayerT,
                                             orc::OrcX86_64> CCMgrT;
      return [](CompileLayerT &CompileLayer, RuntimeDyld::MemoryManager &MemMgr,
                LLVMContext &Context) {
               return llvm::make_unique<CCMgrT>(CompileLayer, MemMgr, Context, 0,
                                                64);
             };
    }
  }
}

OrcCBindingsStack::IndirectStubsManagerBuilder
OrcCBindingsStack::createIndirectStubsMgrBuilder(Triple T) {
  switch (T.getArch()) {
    default: return nullptr;

    case Triple::x86_64:
      return [](){
        return llvm::make_unique<orc::IndirectStubsManager<orc::OrcX86_64>>();
      };
  }
}
