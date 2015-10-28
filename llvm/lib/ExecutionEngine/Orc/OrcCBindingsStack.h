//===--- OrcCBindingsStack.h - Orc JIT stack for C bindings ---*- C++ -*---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_EXECUTIONENGINE_ORC_ORCCBINDINGSSTACK_H
#define LLVM_LIB_EXECUTIONENGINE_ORC_ORCCBINDINGSSTACK_H

#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/IR/LLVMContext.h"

namespace llvm {

class OrcCBindingsStack {
private:

public:

  typedef orc::TargetAddress (*CExternalSymbolResolverFn)(const char *Name,
                                                          void *Ctx);

  typedef orc::JITCompileCallbackManagerBase CompileCallbackMgr;
  typedef orc::ObjectLinkingLayer<> ObjLayerT;
  typedef orc::IRCompileLayer<ObjLayerT> CompileLayerT;
  typedef orc::CompileOnDemandLayer<CompileLayerT, CompileCallbackMgr> CODLayerT;

  typedef std::function<
            std::unique_ptr<CompileCallbackMgr>(CompileLayerT&,
                                                RuntimeDyld::MemoryManager&,
                                                LLVMContext&)>
    CallbackManagerBuilder;

  typedef CODLayerT::IndirectStubsManagerBuilderT IndirectStubsManagerBuilder;

private:

  typedef enum { Invalid, CODLayerHandle, ObjectLayerHandle } HandleType;
  union RawHandleUnion {
    RawHandleUnion() { memset(this, 0, sizeof(RawHandleUnion)); }
    ObjLayerT::ObjSetHandleT Obj;
    CODLayerT::ModuleSetHandleT COD;
  };

  struct ModuleHandleData {

    ModuleHandleData() : Type(Invalid) {}

    ModuleHandleData(ObjLayerT::ObjSetHandleT H)
        : Type(ObjectLayerHandle) {
      RawHandle.Obj = std::move(H);
    }

    ModuleHandleData(CODLayerT::ModuleSetHandleT H)
      : Type(CODLayerHandle) {
      RawHandle.COD = std::move(H);
    }

    HandleType Type;
    RawHandleUnion RawHandle;
  };

public:

  // We need a 'ModuleSetHandleT' to conform to the layer concept.
  typedef unsigned ModuleSetHandleT;

  typedef unsigned ModuleHandleT;

  static CallbackManagerBuilder createCallbackManagerBuilder(Triple T);
  static IndirectStubsManagerBuilder createIndirectStubsMgrBuilder(Triple T);

  OrcCBindingsStack(TargetMachine &TM, LLVMContext &Context,
                    CallbackManagerBuilder &BuildCallbackMgr,
                    IndirectStubsManagerBuilder IndirectStubsMgrBuilder)
    : DL(TM.createDataLayout()),
      ObjectLayer(),
      CompileLayer(ObjectLayer, orc::SimpleCompiler(TM)),
      CCMgr(BuildCallbackMgr(CompileLayer, CCMgrMemMgr, Context)),
      CODLayer(CompileLayer,
               [](Function &F) { std::set<Function*> S; S.insert(&F); return S; },
               *CCMgr, std::move(IndirectStubsMgrBuilder), false),
      CXXRuntimeOverrides([this](const std::string &S) { return mangle(S); }) {}

  ~OrcCBindingsStack() {
    // Run any destructors registered with __cxa_atexit.
    CXXRuntimeOverrides.runDestructors();
    // Run any IR destructors.
    for (auto &DtorRunner : IRStaticDestructorRunners)
      DtorRunner.runViaLayer(*this);
  }

  std::string mangle(StringRef Name) {
    std::string MangledName;
    {
      raw_string_ostream MangledNameStream(MangledName);
      Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    }
    return MangledName;
  }

  template <typename PtrTy>
  static PtrTy fromTargetAddress(orc::TargetAddress Addr) {
    return reinterpret_cast<PtrTy>(static_cast<uintptr_t>(Addr));
  }

  std::shared_ptr<RuntimeDyld::SymbolResolver>
  createResolver(CExternalSymbolResolverFn ExternalResolver,
                 void *ExternalResolverCtx) {
    auto Resolver = orc::createLambdaResolver(
      [this, ExternalResolver, ExternalResolverCtx](const std::string &Name) {
        // Search order:
        // 1. JIT'd symbols.
        // 2. Runtime overrides.
        // 3. External resolver (if present).

        if (auto Sym = CODLayer.findSymbol(Name, true))
          return RuntimeDyld::SymbolInfo(Sym.getAddress(),
                                         Sym.getFlags());
        if (auto Sym = CXXRuntimeOverrides.searchOverrides(Name))
          return Sym;

        if (ExternalResolver)
          return RuntimeDyld::SymbolInfo(ExternalResolver(Name.c_str(),
                                                          ExternalResolverCtx),
                                         llvm::JITSymbolFlags::Exported);

        return RuntimeDyld::SymbolInfo(nullptr);
      },
      [](const std::string &Name) {
        return RuntimeDyld::SymbolInfo(nullptr);
      }
    );

    return std::shared_ptr<RuntimeDyld::SymbolResolver>(std::move(Resolver));
  }

  template <typename LayerT>
  ModuleHandleT addIRModule(LayerT &Layer,
                            Module *M,
                            std::unique_ptr<RuntimeDyld::MemoryManager> MemMgr,
                            CExternalSymbolResolverFn ExternalResolver,
                            void *ExternalResolverCtx) {

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

    // Create the resolver.
    auto Resolver = createResolver(ExternalResolver, ExternalResolverCtx);

    // Add the module to the JIT.
    std::vector<Module*> S;
    S.push_back(std::move(M));

    auto LH = Layer.addModuleSet(std::move(S), std::move(MemMgr),
                                 std::move(Resolver));
    ModuleHandleT H = createHandle(LH);

    // Run the static constructors, and save the static destructor runner for
    // execution when the JIT is torn down.
    orc::CtorDtorRunner<OrcCBindingsStack> CtorRunner(std::move(CtorNames), H);
    CtorRunner.runViaLayer(*this);

    IRStaticDestructorRunners.emplace_back(std::move(DtorNames), H);

    return H;
  }

  ModuleHandleT addIRModuleEager(Module* M,
                                 CExternalSymbolResolverFn ExternalResolver,
                                 void *ExternalResolverCtx) {
    return addIRModule(CompileLayer, std::move(M),
                       llvm::make_unique<SectionMemoryManager>(),
                       std::move(ExternalResolver), ExternalResolverCtx);
  }

  ModuleHandleT addIRModuleLazy(Module* M,
                                CExternalSymbolResolverFn ExternalResolver,
                                void *ExternalResolverCtx) {
    return addIRModule(CODLayer, std::move(M), nullptr,
                       std::move(ExternalResolver), ExternalResolverCtx);
  }

  void removeModule(ModuleHandleT H) {
    auto &HD = HandleData[H];
    switch (HD.Type) {
    case ObjectLayerHandle:
      ObjectLayer.removeObjectSet(HD.RawHandle.Obj);
      break;
    case CODLayerHandle:
      CODLayer.removeModuleSet(HD.RawHandle.COD);
      break;
    default:
      llvm_unreachable("removeModule called on invalid handle type");
    }
  }

  orc::JITSymbol findSymbol(const std::string &Name, bool ExportedSymbolsOnly) {
    return CODLayer.findSymbol(mangle(Name), ExportedSymbolsOnly);
  }

  orc::JITSymbol findSymbolIn(ModuleHandleT H, const std::string &Name,
                              bool ExportedSymbolsOnly) {
    auto &HD = HandleData[H];
    switch (HD.Type) {
    case ObjectLayerHandle:
      return ObjectLayer.findSymbolIn(HD.RawHandle.Obj, mangle(Name),
                                      ExportedSymbolsOnly);
    case CODLayerHandle:
      return CODLayer.findSymbolIn(HD.RawHandle.COD, mangle(Name),
                                   ExportedSymbolsOnly);
    default:
      llvm_unreachable("removeModule called on invalid handle type");
    }
  }

private:

  template <typename LayerHandleT>
  unsigned createHandle(LayerHandleT LH) {
    unsigned NewHandle;
    if (!FreeHandles.empty()) {
      NewHandle = FreeHandles.back();
      FreeHandles.pop_back();
      HandleData[NewHandle] = ModuleHandleData(std::move(LH));
      return NewHandle;
    } else {
      NewHandle = HandleData.size();
      HandleData.push_back(ModuleHandleData(std::move(LH)));
    }
    return NewHandle;
  }

  DataLayout DL;
  SectionMemoryManager CCMgrMemMgr;

  ObjLayerT ObjectLayer;
  CompileLayerT CompileLayer;
  std::unique_ptr<CompileCallbackMgr> CCMgr;
  CODLayerT CODLayer;

  std::vector<ModuleHandleData> HandleData;
  std::vector<unsigned> FreeHandles;

  orc::LocalCXXRuntimeOverrides CXXRuntimeOverrides;
  std::vector<orc::CtorDtorRunner<OrcCBindingsStack>> IRStaticDestructorRunners;
};

} // end namespace llvm

#endif // LLVM_LIB_EXECUTIONENGINE_ORC_ORCCBINDINGSSTACK_H
