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
#include "llvm/Support/DynamicLibrary.h"

using namespace llvm;

OrcLazyJIT::CallbackManagerBuilder
OrcLazyJIT::createCallbackManagerBuilder(Triple T) {
  switch (T.getArch()) {
    default: return nullptr;

    case Triple::x86_64: {
      typedef orc::JITCompileCallbackManager<CompileLayerT,
                                             orc::OrcX86_64> CCMgrT;
      return [](CompileLayerT &CompileLayer, RuntimeDyld::MemoryManager &MemMgr,
                LLVMContext &Context) {
               return make_unique<CCMgrT>(CompileLayer, MemMgr, Context, 0, 64);
             };
    }
  }
}

int llvm::runOrcLazyJIT(std::unique_ptr<Module> M, int ArgC, char* ArgV[]) {
  // Add the program's symbols into the JIT's search space.
  if (sys::DynamicLibrary::LoadLibraryPermanently(nullptr)) {
    errs() << "Error loading program symbols.\n";
    return 1;
  }

  // Grab a target machine and try to build a factory function for the
  // target-specific Orc callback manager.
  auto TM = std::unique_ptr<TargetMachine>(EngineBuilder().selectTarget());
  auto &Context = getGlobalContext();
  auto CallbackMgrBuilder =
    OrcLazyJIT::createCallbackManagerBuilder(Triple(TM->getTargetTriple()));

  // If we couldn't build the factory function then there must not be a callback
  // manager for this target. Bail out.
  if (!CallbackMgrBuilder) {
    errs() << "No callback manager available for target '"
           << TM->getTargetTriple() << "'.\n";
    return 1;
  }

  // Everything looks good. Build the JIT.
  OrcLazyJIT J(std::move(TM), Context, CallbackMgrBuilder);

  // Add the module, look up main and run it.
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
