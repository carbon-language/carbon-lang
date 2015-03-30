//===------ OrcLazyJIT.cpp - Basic Orc-based JIT for lazy execution -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OrcLazyJIT.h"
#include "llvm/ExecutionEngine/Orc/OrcTargetSupport.h"

using namespace llvm;

std::unique_ptr<OrcLazyJIT::CompileCallbackMgr>
OrcLazyJIT::createCallbackMgr(Triple T, LLVMContext &Context) {
  switch (T.getArch()) {
    default:
      // Flag error.
      Error = true;
      return nullptr;

    case Triple::x86_64: {
      typedef orc::JITCompileCallbackManager<CompileLayerT,
                                             orc::OrcX86_64> CCMgrT;
      return make_unique<CCMgrT>(CompileLayer, CCMgrMemMgr, Context, 0, 64);
    }
  }
}

int llvm::runOrcLazyJIT(std::unique_ptr<Module> M, int ArgC, char* ArgV[]) {
  OrcLazyJIT J(std::unique_ptr<TargetMachine>(EngineBuilder().selectTarget()),
               getGlobalContext());

  if (!J.Ok()) {
    errs() << "Could not construct JIT.\n";
    return 1;
  }

  auto MainHandle = J.addModule(std::move(M));
  auto MainSym = J.findSymbolIn(MainHandle, "main");

  if (!MainSym) {
    errs() << "Could not find main function.\n";
    return 1;
  }

  typedef int (*MainFnPtr)(int, char*[]);
  auto Main = reinterpret_cast<MainFnPtr>(
                static_cast<uintptr_t>(MainSym.getAddress()));

  return Main(ArgC, ArgV);
}
