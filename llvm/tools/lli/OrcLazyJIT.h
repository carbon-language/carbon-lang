//===--- OrcLazyJIT.h - Basic Orc-based JIT for lazy execution --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Simple Orc-based JIT. Uses the compile-on-demand layer to break up and
// lazily compile modules.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLI_ORCLAZYJIT_H
#define LLVM_TOOLS_LLI_ORCLAZYJIT_H

#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/IR/LLVMContext.h"

namespace llvm {

class OrcLazyJIT {
public:

  typedef orc::JITCompileCallbackManagerBase CompileCallbackMgr;
  typedef orc::ObjectLinkingLayer<> ObjLayerT;
  typedef orc::IRCompileLayer<ObjLayerT> CompileLayerT;
  typedef std::function<std::unique_ptr<Module>(std::unique_ptr<Module>)>
    TransformFtor;
  typedef orc::IRTransformLayer<CompileLayerT, TransformFtor> IRDumpLayerT;
  typedef orc::CompileOnDemandLayer<IRDumpLayerT, CompileCallbackMgr> CODLayerT;
  typedef CODLayerT::ModuleSetHandleT ModuleHandleT;

  typedef std::function<
            std::unique_ptr<CompileCallbackMgr>(IRDumpLayerT&,
                                                RuntimeDyld::MemoryManager&,
                                                LLVMContext&)>
    CallbackManagerBuilder;

  static CallbackManagerBuilder createCallbackManagerBuilder(Triple T);
  const DataLayout &DL;

  OrcLazyJIT(std::unique_ptr<TargetMachine> TM, const DataLayout &DL,
             LLVMContext &Context, CallbackManagerBuilder &BuildCallbackMgr)
      : DL(DL), TM(std::move(TM)), ObjectLayer(),
        CompileLayer(ObjectLayer, orc::SimpleCompiler(*this->TM)),
        IRDumpLayer(CompileLayer, createDebugDumper()),
        CCMgr(BuildCallbackMgr(IRDumpLayer, CCMgrMemMgr, Context)),
        CODLayer(IRDumpLayer, *CCMgr, extractSingleFunction, false),
        CXXRuntimeOverrides(
            [this](const std::string &S) { return mangle(S); }) {}

  ~OrcLazyJIT() {
    // Run any destructors registered with __cxa_atexit.
    CXXRuntimeOverrides.runDestructors();
    // Run any IR destructors.
    for (auto &DtorRunner : IRStaticDestructorRunners)
      DtorRunner.runViaLayer(CODLayer);
  }

  template <typename PtrTy>
  static PtrTy fromTargetAddress(orc::TargetAddress Addr) {
    return reinterpret_cast<PtrTy>(static_cast<uintptr_t>(Addr));
  }

  ModuleHandleT addModule(std::unique_ptr<Module> M) {
    // Attach a data-layout if one isn't already present.
    if (M->getDataLayout().isDefault())
      M->setDataLayout(DL);

    // Record the static constructors and destructors. We have to do this before
    // we hand over ownership of the module to the JIT.
    std::vector<std::string> CtorNames, DtorNames;
    for (auto Ctor : orc::getConstructors(*M))
      CtorNames.push_back(mangle(Ctor.Func->getName()));
    for (auto Dtor : orc::getDestructors(*M))
      DtorNames.push_back(mangle(Dtor.Func->getName()));

    // Symbol resolution order:
    //   1) Search the JIT symbols.
    //   2) Check for C++ runtime overrides.
    //   3) Search the host process (LLI)'s symbol table.
    std::shared_ptr<RuntimeDyld::SymbolResolver> Resolver =
      orc::createLambdaResolver(
        [this](const std::string &Name) {
          if (auto Sym = CODLayer.findSymbol(Name, true))
            return RuntimeDyld::SymbolInfo(Sym.getAddress(),
                                           Sym.getFlags());
          if (auto Sym = CXXRuntimeOverrides.searchOverrides(Name))
            return Sym;

          if (auto Addr =
              RTDyldMemoryManager::getSymbolAddressInProcess(Name))
            return RuntimeDyld::SymbolInfo(Addr, JITSymbolFlags::Exported);

          return RuntimeDyld::SymbolInfo(nullptr);
        },
        [](const std::string &Name) {
          return RuntimeDyld::SymbolInfo(nullptr);
        }
      );

    // Add the module to the JIT.
    std::vector<std::unique_ptr<Module>> S;
    S.push_back(std::move(M));
    auto H = CODLayer.addModuleSet(std::move(S), nullptr, std::move(Resolver));

    // Run the static constructors, and save the static destructor runner for
    // execution when the JIT is torn down.
    orc::CtorDtorRunner<CODLayerT> CtorRunner(std::move(CtorNames), H);
    CtorRunner.runViaLayer(CODLayer);

    IRStaticDestructorRunners.emplace_back(std::move(DtorNames), H);

    return H;
  }

  orc::JITSymbol findSymbol(const std::string &Name) {
    return CODLayer.findSymbol(mangle(Name), true);
  }

  orc::JITSymbol findSymbolIn(ModuleHandleT H, const std::string &Name) {
    return CODLayer.findSymbolIn(H, mangle(Name), true);
  }

private:

  std::string mangle(const std::string &Name) {
    std::string MangledName;
    {
      raw_string_ostream MangledNameStream(MangledName);
      Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    }
    return MangledName;
  }

  static std::set<Function*> extractSingleFunction(Function &F) {
    std::set<Function*> Partition;
    Partition.insert(&F);
    return Partition;
  }

  static TransformFtor createDebugDumper();

  std::unique_ptr<TargetMachine> TM;
  SectionMemoryManager CCMgrMemMgr;

  ObjLayerT ObjectLayer;
  CompileLayerT CompileLayer;
  IRDumpLayerT IRDumpLayer;
  std::unique_ptr<CompileCallbackMgr> CCMgr;
  CODLayerT CODLayer;

  orc::LocalCXXRuntimeOverrides CXXRuntimeOverrides;
  std::vector<orc::CtorDtorRunner<CODLayerT>> IRStaticDestructorRunners;
};

int runOrcLazyJIT(std::unique_ptr<Module> M, int ArgC, char* ArgV[]);

} // end namespace llvm

#endif
