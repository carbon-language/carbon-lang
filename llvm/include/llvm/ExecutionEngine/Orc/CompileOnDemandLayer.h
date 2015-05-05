//===- CompileOnDemandLayer.h - Compile each function on demand -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// JIT layer for breaking up modules and inserting callbacks to allow
// individual functions to be compiled on demand.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_COMPILEONDEMANDLAYER_H
#define LLVM_EXECUTIONENGINE_ORC_COMPILEONDEMANDLAYER_H

//#include "CloneSubModule.h"
#include "IndirectionUtils.h"
#include "LambdaResolver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <list>
#include <set>

#include "llvm/Support/Debug.h"

namespace llvm {
namespace orc {

/// @brief Compile-on-demand layer.
///
///   When a module is added to this layer a stub is created for each of its
/// function definitions. The stubs and other global values are immediately
/// added to the layer below. When a stub is called it triggers the extraction
/// of the function body from the original module. The extracted body is then
/// compiled and executed.
template <typename BaseLayerT, typename CompileCallbackMgrT>
class CompileOnDemandLayer {
private:

  // Utility class for MapValue. Only materializes declarations for global
  // variables.
  class GlobalDeclMaterializer : public ValueMaterializer {
  public:
    GlobalDeclMaterializer(Module &Dst) : Dst(Dst) {}
    Value* materializeValueFor(Value *V) final {
      if (auto *GV = dyn_cast<GlobalVariable>(V))
        return cloneGlobalVariableDecl(Dst, *GV);
      else if (auto *F = dyn_cast<Function>(V))
        return cloneFunctionDecl(Dst, *F);
      // Else.
      return nullptr;
    }
  private:
    Module &Dst;
  };

  typedef typename BaseLayerT::ModuleSetHandleT BaseLayerModuleSetHandleT;
  class UncompiledPartition;

  // Logical module.
  //
  //   This struct contains the handles for the global values and stubs (which
  // cover the external symbols of the original module), plus the handes for
  // each of the extracted partitions. These handleds are used for lookup (only
  // the globals/stubs module is searched) and memory management. The actual
  // searching and resource management are handled by the LogicalDylib that owns
  // the LogicalModule.
  struct LogicalModule {
    LogicalModule() {}

    LogicalModule(LogicalModule &&Other)
        : SrcM(std::move(Other.SrcM)),
          GVsAndStubsHandle(std::move(Other.GVsAndStubsHandle)),
          ImplHandles(std::move(Other.ImplHandles)) {}

    std::unique_ptr<Module> SrcM;
    BaseLayerModuleSetHandleT GVsAndStubsHandle;
    std::vector<BaseLayerModuleSetHandleT> ImplHandles;
  };

  // Logical dylib.
  //
  //   This class handles symbol resolution and resource management for a set of
  // modules that were added together as a logical dylib.
  //
  //   A logical dylib contains one-or-more LogicalModules plus a set of
  // UncompiledPartitions. LogicalModules support symbol resolution and resource
  // management for for code that has already been emitted. UncompiledPartitions
  // represent code that has not yet been compiled.
  class LogicalDylib {
  private:
    friend class UncompiledPartition;
    typedef std::list<LogicalModule> LogicalModuleList;
  public:

    typedef unsigned UncompiledPartitionID;
    typedef typename LogicalModuleList::iterator LMHandle;

    // Construct a logical dylib.
    LogicalDylib(CompileOnDemandLayer &CODLayer) : CODLayer(CODLayer) { }

    // Delete this logical dylib, release logical module resources.
    virtual ~LogicalDylib() {
      releaseLogicalModuleResources();
    }

    // Get a reference to the containing layer.
    CompileOnDemandLayer& getCODLayer() { return CODLayer; }

    // Get a reference to the base layer.
    BaseLayerT& getBaseLayer() { return CODLayer.BaseLayer; }

    // Start a new context for a single logical module.
    LMHandle createLogicalModule() {
      LogicalModules.push_back(LogicalModule());
      return std::prev(LogicalModules.end());
    }

    // Set the global-values-and-stubs module handle for this logical module.
    void setGVsAndStubsHandle(LMHandle LMH, BaseLayerModuleSetHandleT H) {
      LMH->GVsAndStubsHandle = H;
    }

    // Return the global-values-and-stubs module handle for this logical module.
    BaseLayerModuleSetHandleT getGVsAndStubsHandle(LMHandle LMH) {
      return LMH->GVsAndStubsHandle;
    }

    //   Add a handle to a module containing lazy function bodies to the given
    // logical module.
    void addToLogicalModule(LMHandle LMH, BaseLayerModuleSetHandleT H) {
      LMH->ImplHandles.push_back(H);
    }

    // Create an UncompiledPartition attached to this LogicalDylib.
    UncompiledPartition& createUncompiledPartition(LMHandle LMH,
                                                   std::shared_ptr<Module> SrcM);

    // Take ownership of the given UncompiledPartition from the logical dylib.
    std::unique_ptr<UncompiledPartition>
    takeUPOwnership(UncompiledPartitionID ID);

    // Look up a symbol in this context.
    JITSymbol findSymbolInternally(LMHandle LMH, const std::string &Name) {
      if (auto Symbol = getBaseLayer().findSymbolIn(LMH->GVsAndStubsHandle,
                                                    Name, false))
        return Symbol;

      for (auto I = LogicalModules.begin(), E = LogicalModules.end(); I != E;
           ++I)
        if (I != LMH)
          if (auto Symbol = getBaseLayer().findSymbolIn(I->GVsAndStubsHandle,
                                                        Name, false))
            return Symbol;

      return nullptr;
    }

    JITSymbol findSymbol(const std::string &Name, bool ExportedSymbolsOnly) {
      for (auto &LM : LogicalModules)
        if (auto Symbol = getBaseLayer().findSymbolIn(LM.GVsAndStubsHandle,
                                                      Name,
                                                      ExportedSymbolsOnly))
          return Symbol;
      return nullptr;
    }

    // Find an external symbol (via the user supplied SymbolResolver).
    virtual RuntimeDyld::SymbolInfo
    findSymbolExternally(const std::string &Name) const = 0;

  private:

    void releaseLogicalModuleResources() {
      for (auto I = LogicalModules.begin(), E = LogicalModules.end(); I != E;
           ++I) {
        getBaseLayer().removeModuleSet(I->GVsAndStubsHandle);
        for (auto H : I->ImplHandles)
          getBaseLayer().removeModuleSet(H);
      }
    }

    CompileOnDemandLayer &CODLayer;
    LogicalModuleList LogicalModules;
    std::vector<std::unique_ptr<UncompiledPartition>> UncompiledPartitions;
  };

  template <typename ResolverPtrT>
  class LogicalDylibImpl : public LogicalDylib  {
  public:
    LogicalDylibImpl(CompileOnDemandLayer &CODLayer, ResolverPtrT Resolver)
      : LogicalDylib(CODLayer), Resolver(std::move(Resolver)) {}

    RuntimeDyld::SymbolInfo
    findSymbolExternally(const std::string &Name) const override {
      return Resolver->findSymbol(Name);
    }

  private:
    ResolverPtrT Resolver;
  };

  template <typename ResolverPtrT>
  static std::unique_ptr<LogicalDylib>
  createLogicalDylib(CompileOnDemandLayer &CODLayer,
                     ResolverPtrT Resolver) {
    typedef LogicalDylibImpl<ResolverPtrT> Impl;
    return llvm::make_unique<Impl>(CODLayer, std::move(Resolver));
  }

  // Uncompiled partition.
  //
  // Represents one as-yet uncompiled portion of a module.
  class UncompiledPartition {
  public:

    struct PartitionEntry {
      PartitionEntry(Function *F, TargetAddress CallbackID)
          : F(F), CallbackID(CallbackID) {}
      Function *F;
      TargetAddress CallbackID;
    };

    typedef std::vector<PartitionEntry> PartitionEntryList;

    // Creates an uncompiled partition with the list of functions that make up
    // this partition.
    UncompiledPartition(LogicalDylib &LD, typename LogicalDylib::LMHandle LMH,
                        std::shared_ptr<Module> SrcM)
        : LD(LD), LMH(LMH), SrcM(std::move(SrcM)), ID(~0U) {}

    ~UncompiledPartition() {
      // FIXME: When we want to support threaded lazy compilation we'll need to
      //        lock the callback manager here.
      auto &CCMgr = LD.getCODLayer().CompileCallbackMgr;
      for (auto PEntry : PartitionEntries)
        CCMgr.releaseCompileCallback(PEntry.CallbackID);
    }

    // Set the ID for this partition.
    void setID(typename LogicalDylib::UncompiledPartitionID ID) {
      this->ID = ID;
    }

    // Set the function set and callbacks for this partition.
    void setPartitionEntries(PartitionEntryList PartitionEntries) {
      this->PartitionEntries = std::move(PartitionEntries);
    }

    // Handle a compile callback for the function at index FnIdx.
    TargetAddress compile(unsigned FnIdx) {
      // Take ownership of self. This will ensure we delete the partition and
      // free all its resources once we're done compiling.
      std::unique_ptr<UncompiledPartition> This = LD.takeUPOwnership(ID);

      // Release all other compile callbacks for this partition.
      // We skip the callback for this function because that's the one that
      // called us, and the callback manager will already have removed it.
      auto &CCMgr = LD.getCODLayer().CompileCallbackMgr;
      for (unsigned I = 0; I < PartitionEntries.size(); ++I)
        if (I != FnIdx)
          CCMgr.releaseCompileCallback(PartitionEntries[I].CallbackID);

      // Grab the name of the function being called here.
      Function *F = PartitionEntries[FnIdx].F;
      std::string CalledFnName = Mangle(F->getName(), SrcM->getDataLayout());

      // Extract the function and add it to the base layer.
      auto PartitionImplH = emitPartition();
      LD.addToLogicalModule(LMH, PartitionImplH);

      // Update body pointers.
      // FIXME: When we start supporting remote lazy jitting this will need to
      //        be replaced with a user-supplied callback for updating the
      //        remote pointers.
      TargetAddress CalledAddr = 0;
      for (unsigned I = 0; I < PartitionEntries.size(); ++I) {
        auto F = PartitionEntries[I].F;
        std::string FName(F->getName());
        auto FnBodySym =
          LD.getBaseLayer().findSymbolIn(PartitionImplH,
                                         Mangle(FName, SrcM->getDataLayout()),
                                         false);
        auto FnPtrSym =
          LD.getBaseLayer().findSymbolIn(LD.getGVsAndStubsHandle(LMH),
                                         Mangle(FName + "$orc_addr",
                                                SrcM->getDataLayout()),
                                         false);
        assert(FnBodySym && "Couldn't find function body.");
        assert(FnPtrSym && "Couldn't find function body pointer.");

        auto FnBodyAddr = FnBodySym.getAddress();
        void *FnPtrAddr = reinterpret_cast<void*>(
                            static_cast<uintptr_t>(FnPtrSym.getAddress()));

        // If this is the function we're calling record the address so we can
        // return it from this function.
        if (I == FnIdx)
          CalledAddr = FnBodyAddr;

        memcpy(FnPtrAddr, &FnBodyAddr, sizeof(uintptr_t));
      }

      // Finally, clear the partition structure so we don't try to
      // double-release the callbacks in the UncompiledPartition destructor.
      PartitionEntries.clear();

      return CalledAddr;
    }

  private:

    BaseLayerModuleSetHandleT emitPartition() {
      // Create the module.
      std::string NewName(SrcM->getName());
      for (auto &PEntry : PartitionEntries) {
        NewName += ".";
        NewName += PEntry.F->getName();
      }
      auto PM = llvm::make_unique<Module>(NewName, SrcM->getContext());
      PM->setDataLayout(SrcM->getDataLayout());
      ValueToValueMapTy VMap;
      GlobalDeclMaterializer GDM(*PM);

      // Create decls in the new module.
      for (auto &PEntry : PartitionEntries)
        cloneFunctionDecl(*PM, *PEntry.F, &VMap);

      // Move the function bodies.
      for (auto &PEntry : PartitionEntries)
        moveFunctionBody(*PEntry.F, VMap);

      // Create memory manager and symbol resolver.
      auto MemMgr = llvm::make_unique<SectionMemoryManager>();
      auto Resolver = createLambdaResolver(
          [this](const std::string &Name) {
            if (auto Symbol = LD.findSymbolInternally(LMH, Name))
              return RuntimeDyld::SymbolInfo(Symbol.getAddress(),
                                             Symbol.getFlags());
            return LD.findSymbolExternally(Name);
          },
          [this](const std::string &Name) {
            if (auto Symbol = LD.findSymbolInternally(LMH, Name))
              return RuntimeDyld::SymbolInfo(Symbol.getAddress(),
                                             Symbol.getFlags());
            return RuntimeDyld::SymbolInfo(nullptr);
          });
      std::vector<std::unique_ptr<Module>> PartMSet;
      PartMSet.push_back(std::move(PM));
      return LD.getBaseLayer().addModuleSet(std::move(PartMSet),
                                            std::move(MemMgr),
                                            std::move(Resolver));
    }

    LogicalDylib &LD;
    typename LogicalDylib::LMHandle LMH;
    std::shared_ptr<Module> SrcM;
    typename LogicalDylib::UncompiledPartitionID ID;
    PartitionEntryList PartitionEntries;
  };

  typedef std::list<std::unique_ptr<LogicalDylib>> LogicalDylibList;

public:
  /// @brief Handle to a set of loaded modules.
  typedef typename LogicalDylibList::iterator ModuleSetHandleT;

  /// @brief Construct a compile-on-demand layer instance.
  CompileOnDemandLayer(BaseLayerT &BaseLayer, CompileCallbackMgrT &CallbackMgr)
      : BaseLayer(BaseLayer), CompileCallbackMgr(CallbackMgr) {}

  /// @brief Add a module to the compile-on-demand layer.
  template <typename ModuleSetT, typename MemoryManagerPtrT,
            typename SymbolResolverPtrT>
  ModuleSetHandleT addModuleSet(ModuleSetT Ms,
                                MemoryManagerPtrT MemMgr,
                                SymbolResolverPtrT Resolver) {

    assert(MemMgr == nullptr &&
           "User supplied memory managers not supported with COD yet.");

    LogicalDylibs.push_back(createLogicalDylib(*this, std::move(Resolver)));

    // Process each of the modules in this module set.
    for (auto &M : Ms) {
      std::vector<std::vector<Function*>> Partitioning;
      for (auto &F : *M) {
        if (F.isDeclaration())
          continue;
        Partitioning.push_back(std::vector<Function*>());
        Partitioning.back().push_back(&F);
      }
      addLogicalModule(*LogicalDylibs.back(),
                       std::shared_ptr<Module>(std::move(M)),
                       std::move(Partitioning));
    }

    return std::prev(LogicalDylibs.end());
  }

  /// @brief Remove the module represented by the given handle.
  ///
  ///   This will remove all modules in the layers below that were derived from
  /// the module represented by H.
  void removeModuleSet(ModuleSetHandleT H) {
    LogicalDylibs.erase(H);
  }

  /// @brief Search for the given named symbol.
  /// @param Name The name of the symbol to search for.
  /// @param ExportedSymbolsOnly If true, search only for exported symbols.
  /// @return A handle for the given named symbol, if it exists.
  JITSymbol findSymbol(StringRef Name, bool ExportedSymbolsOnly) {
    return BaseLayer.findSymbol(Name, ExportedSymbolsOnly);
  }

  /// @brief Get the address of a symbol provided by this layer, or some layer
  ///        below this one.
  JITSymbol findSymbolIn(ModuleSetHandleT H, const std::string &Name,
                         bool ExportedSymbolsOnly) {
    return (*H)->findSymbol(Name, ExportedSymbolsOnly);
  }

private:

  void addLogicalModule(LogicalDylib &LD, std::shared_ptr<Module> SrcM,
                        std::vector<std::vector<Function*>> Partitions) {

    // Bump the linkage and rename any anonymous/privote members in SrcM to
    // ensure that everything will resolve properly after we partition SrcM.
    makeAllSymbolsExternallyAccessible(*SrcM);

    // Create a logical module handle for SrcM within the logical dylib.
    auto LMH = LD.createLogicalModule();

    // Create the GVs-and-stubs module.
    auto GVsAndStubsM = llvm::make_unique<Module>(
                          (SrcM->getName() + ".globals_and_stubs").str(),
                          SrcM->getContext());
    GVsAndStubsM->setDataLayout(SrcM->getDataLayout());
    ValueToValueMapTy VMap;

    // Process partitions and create stubs.
    // We create the stubs before copying the global variables as we know the
    // stubs won't refer to any globals (they only refer to their implementation
    // pointer) so there's no ordering/value-mapping issues.
    for (auto& Partition : Partitions) {
      auto &UP = LD.createUncompiledPartition(LMH, SrcM);
      typename UncompiledPartition::PartitionEntryList PartitionEntries;
      for (auto &F : Partition) {
        assert(!F->isDeclaration() &&
               "Partition should only contain definitions");
        unsigned FnIdx = PartitionEntries.size();
        auto CCI = CompileCallbackMgr.getCompileCallback(SrcM->getContext());
        PartitionEntries.push_back(
          typename UncompiledPartition::PartitionEntry(F, CCI.getAddress()));
        Function *StubF = cloneFunctionDecl(*GVsAndStubsM, *F, &VMap);
        GlobalVariable *FnBodyPtr =
          createImplPointer(*StubF->getType(), *StubF->getParent(),
                            StubF->getName() + "$orc_addr",
                            createIRTypedAddress(*StubF->getFunctionType(),
                                                 CCI.getAddress()));
        makeStub(*StubF, *FnBodyPtr);
        CCI.setCompileAction([&UP, FnIdx]() { return UP.compile(FnIdx); });
      }

      UP.setPartitionEntries(std::move(PartitionEntries));
    }

    // Now clone the global variable declarations.
    GlobalDeclMaterializer GDMat(*GVsAndStubsM);
    for (auto &GV : SrcM->globals())
      if (!GV.isDeclaration())
        cloneGlobalVariableDecl(*GVsAndStubsM, GV, &VMap);

    // Then clone the initializers.
    for (auto &GV : SrcM->globals())
      if (!GV.isDeclaration())
        moveGlobalVariableInitializer(GV, VMap, &GDMat);

    // Build a resolver for the stubs module and add it to the base layer.
    auto GVsAndStubsResolver = createLambdaResolver(
        [&LD](const std::string &Name) {
          if (auto Symbol = LD.findSymbol(Name, false))
            return RuntimeDyld::SymbolInfo(Symbol.getAddress(),
                                           Symbol.getFlags());
          return LD.findSymbolExternally(Name);
        },
        [&LD](const std::string &Name) {
          return RuntimeDyld::SymbolInfo(nullptr);
        });

    std::vector<std::unique_ptr<Module>> GVsAndStubsMSet;
    GVsAndStubsMSet.push_back(std::move(GVsAndStubsM));
    auto GVsAndStubsH =
      BaseLayer.addModuleSet(std::move(GVsAndStubsMSet),
                             llvm::make_unique<SectionMemoryManager>(),
                             std::move(GVsAndStubsResolver));
    LD.setGVsAndStubsHandle(LMH, GVsAndStubsH);
  }

  static std::string Mangle(StringRef Name, const DataLayout &DL) {
    Mangler M(&DL);
    std::string MangledName;
    {
      raw_string_ostream MangledNameStream(MangledName);
      M.getNameWithPrefix(MangledNameStream, Name);
    }
    return MangledName;
  }

  BaseLayerT &BaseLayer;
  CompileCallbackMgrT &CompileCallbackMgr;
  LogicalDylibList LogicalDylibs;
};

template <typename BaseLayerT, typename CompileCallbackMgrT>
typename CompileOnDemandLayer<BaseLayerT, CompileCallbackMgrT>::
           UncompiledPartition&
CompileOnDemandLayer<BaseLayerT, CompileCallbackMgrT>::LogicalDylib::
  createUncompiledPartition(LMHandle LMH, std::shared_ptr<Module> SrcM) {
  UncompiledPartitions.push_back(
      llvm::make_unique<UncompiledPartition>(*this, LMH, std::move(SrcM)));
  UncompiledPartitions.back()->setID(UncompiledPartitions.size() - 1);
  return *UncompiledPartitions.back();
}

template <typename BaseLayerT, typename CompileCallbackMgrT>
std::unique_ptr<typename CompileOnDemandLayer<BaseLayerT, CompileCallbackMgrT>::
                  UncompiledPartition>
CompileOnDemandLayer<BaseLayerT, CompileCallbackMgrT>::LogicalDylib::
  takeUPOwnership(UncompiledPartitionID ID) {

  std::swap(UncompiledPartitions[ID], UncompiledPartitions.back());
  UncompiledPartitions[ID]->setID(ID);
  auto UP = std::move(UncompiledPartitions.back());
  UncompiledPartitions.pop_back();
  return UP;
}

} // End namespace orc.
} // End namespace llvm.

#endif // LLVM_EXECUTIONENGINE_ORC_COMPILEONDEMANDLAYER_H
