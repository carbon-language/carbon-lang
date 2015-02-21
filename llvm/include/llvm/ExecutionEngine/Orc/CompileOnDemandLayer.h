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

#include "IndirectionUtils.h"
#include "LookasideRTDyldMM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include <list>

namespace llvm {
namespace orc {

/// @brief Compile-on-demand layer.
///
///   Modules added to this layer have their calls indirected, and are then
/// broken up into a set of single-function modules, each of which is added
/// to the layer below in a singleton set. The lower layer can be any layer that
/// accepts IR module sets.
///
/// It is expected that this layer will frequently be used on top of a
/// LazyEmittingLayer. The combination of the two ensures that each function is
/// compiled only when it is first called.
template <typename BaseLayerT, typename CompileCallbackMgrT>
class CompileOnDemandLayer {
public:
  /// @brief Lookup helper that provides compatibility with the classic
  ///        static-compilation symbol resolution process.
  ///
  ///   The CompileOnDemand (COD) layer splits modules up into multiple
  /// sub-modules, each held in its own llvm::Module instance, in order to
  /// support lazy compilation. When a module that contains private symbols is
  /// broken up symbol linkage changes may be required to enable access to
  /// "private" data that now resides in a different llvm::Module instance. To
  /// retain expected symbol resolution behavior for clients of the COD layer,
  /// the CODScopedLookup class uses a two-tiered lookup system to resolve
  /// symbols. Lookup first scans sibling modules that were split from the same
  /// original module (logical-module scoped lookup), then scans all other
  /// modules that have been added to the lookup scope (logical-dylib scoped
  /// lookup).
  class CODScopedLookup {
  private:
    typedef typename BaseLayerT::ModuleSetHandleT BaseLayerModuleSetHandleT;
    typedef std::vector<BaseLayerModuleSetHandleT> SiblingHandlesList;
    typedef std::list<SiblingHandlesList> PseudoDylibModuleSetHandlesList;

  public:
    /// @brief Handle for a logical module.
    typedef typename PseudoDylibModuleSetHandlesList::iterator LMHandle;

    /// @brief Construct a scoped lookup.
    CODScopedLookup(BaseLayerT &BaseLayer) : BaseLayer(BaseLayer) {}

    /// @brief Start a new context for a single logical module.
    LMHandle createLogicalModule() {
      Handles.push_back(SiblingHandlesList());
      return std::prev(Handles.end());
    }

    /// @brief Add a concrete Module's handle to the given logical Module's
    ///        lookup scope.
    void addToLogicalModule(LMHandle LMH, BaseLayerModuleSetHandleT H) {
      LMH->push_back(H);
    }

    /// @brief Remove a logical Module from the CODScopedLookup entirely.
    void removeLogicalModule(LMHandle LMH) { Handles.erase(LMH); }

    /// @brief Look up a symbol in this context.
    JITSymbol findSymbol(LMHandle LMH, const std::string &Name) {
      if (auto Symbol = findSymbolIn(LMH, Name))
        return Symbol;

      for (auto I = Handles.begin(), E = Handles.end(); I != E; ++I)
        if (I != LMH)
          if (auto Symbol = findSymbolIn(I, Name))
            return Symbol;

      return nullptr;
    }

  private:

    JITSymbol findSymbolIn(LMHandle LMH, const std::string &Name) {
      for (auto H : *LMH)
        if (auto Symbol = BaseLayer.findSymbolIn(H, Name, false))
          return Symbol;
      return nullptr;
    }

    BaseLayerT &BaseLayer;
    PseudoDylibModuleSetHandlesList Handles;
  };

private:
  typedef typename BaseLayerT::ModuleSetHandleT BaseLayerModuleSetHandleT;
  typedef std::vector<BaseLayerModuleSetHandleT> BaseLayerModuleSetHandleListT;

  struct ModuleSetInfo {
    // Symbol lookup - just one for the whole module set.
    std::shared_ptr<CODScopedLookup> Lookup;

    // Logical module handles.
    std::vector<typename CODScopedLookup::LMHandle> LMHandles;

    // List of vectors of module set handles:
    // One vector per logical module - each vector holds the handles for the
    // exploded modules for that logical module in the base layer.
    BaseLayerModuleSetHandleListT BaseLayerModuleSetHandles;

    ModuleSetInfo(std::shared_ptr<CODScopedLookup> Lookup)
        : Lookup(std::move(Lookup)) {}

    void releaseResources(BaseLayerT &BaseLayer) {
      for (auto LMH : LMHandles)
        Lookup->removeLogicalModule(LMH);
      for (auto H : BaseLayerModuleSetHandles)
        BaseLayer.removeModuleSet(H);
    }
  };

  typedef std::list<ModuleSetInfo> ModuleSetInfoListT;

public:
  /// @brief Handle to a set of loaded modules.
  typedef typename ModuleSetInfoListT::iterator ModuleSetHandleT;

  // @brief Fallback lookup functor.
  typedef std::function<uint64_t(const std::string &)> LookupFtor;

  /// @brief Construct a compile-on-demand layer instance.
  CompileOnDemandLayer(BaseLayerT &BaseLayer, LLVMContext &Context)
    : BaseLayer(BaseLayer),
      CompileCallbackMgr(BaseLayer, Context, 0, 64) {}

  /// @brief Add a module to the compile-on-demand layer.
  template <typename ModuleSetT>
  ModuleSetHandleT addModuleSet(ModuleSetT Ms,
                                LookupFtor FallbackLookup = nullptr) {

    // If the user didn't supply a fallback lookup then just use
    // getSymbolAddress.
    if (!FallbackLookup)
      FallbackLookup = [=](const std::string &Name) {
                         return findSymbol(Name, true).getAddress();
                       };

    // Create a lookup context and ModuleSetInfo for this module set.
    // For the purposes of symbol resolution the set Ms will be treated as if
    // the modules it contained had been linked together as a dylib.
    auto DylibLookup = std::make_shared<CODScopedLookup>(BaseLayer);
    ModuleSetHandleT H =
        ModuleSetInfos.insert(ModuleSetInfos.end(), ModuleSetInfo(DylibLookup));
    ModuleSetInfo &MSI = ModuleSetInfos.back();

    // Process each of the modules in this module set.
    for (auto &M : Ms)
      partitionAndAdd(*M, MSI, FallbackLookup);

    return H;
  }

  /// @brief Remove the module represented by the given handle.
  ///
  ///   This will remove all modules in the layers below that were derived from
  /// the module represented by H.
  void removeModuleSet(ModuleSetHandleT H) {
    H->releaseResources(BaseLayer);
    ModuleSetInfos.erase(H);
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
    BaseLayerModuleSetHandleListT &BaseLayerHandles = H->second;
    for (auto &BH : BaseLayerHandles) {
      if (auto Symbol = BaseLayer.findSymbolIn(BH, Name, ExportedSymbolsOnly))
        return Symbol;
    }
    return nullptr;
  }

private:

  void partitionAndAdd(Module &M, ModuleSetInfo &MSI,
                       LookupFtor FallbackLookup) {
    const char *AddrSuffix = "$orc_addr";
    const char *BodySuffix = "$orc_body";

    // We're going to break M up into a bunch of sub-modules, but we want
    // internal linkage symbols to still resolve sensibly. CODScopedLookup
    // provides the "logical module" concept to make this work, so create a
    // new logical module for M.
    auto DylibLookup = MSI.Lookup;
    auto LogicalModule = DylibLookup->createLogicalModule();
    MSI.LMHandles.push_back(LogicalModule);

    // Partition M into a "globals and stubs" module, a "common symbols" module,
    // and a list of single-function modules.
    auto PartitionedModule = fullyPartition(M);
    auto StubsModule = std::move(PartitionedModule.GlobalVars);
    auto CommonsModule = std::move(PartitionedModule.Commons);
    auto FunctionModules = std::move(PartitionedModule.Functions);

    // Emit the commons stright away.
    auto CommonHandle = addModule(std::move(CommonsModule), MSI, LogicalModule,
                                  FallbackLookup);
    BaseLayer.emitAndFinalize(CommonHandle);

    // Map of definition names to callback-info data structures. We'll use
    // this to build the compile actions for the stubs below.
    typedef std::map<std::string,
                     typename CompileCallbackMgrT::CompileCallbackInfo>
      StubInfoMap;
    StubInfoMap StubInfos;

    // Now we need to take each of the extracted Modules and add them to
    // base layer. Each Module will be added individually to make sure they
    // can be compiled separately, and each will get its own lookaside
    // memory manager that will resolve within this logical module first.
    for (auto &SubM : FunctionModules) {

      // Keep track of the stubs we create for this module so that we can set
      // their compile actions.
      std::vector<typename StubInfoMap::iterator> NewStubInfos;

      // Search for function definitions and insert stubs into the stubs
      // module.
      for (auto &F : *SubM) {
        if (F.isDeclaration())
          continue;

        std::string Name = F.getName();
        Function *Proto = StubsModule->getFunction(Name);
        assert(Proto && "Failed to clone function decl into stubs module.");
        auto CallbackInfo =
          CompileCallbackMgr.getCompileCallback(*Proto->getFunctionType());
        GlobalVariable *FunctionBodyPointer =
          createImplPointer(*Proto, Name + AddrSuffix,
                            CallbackInfo.getAddress());
        makeStub(*Proto, *FunctionBodyPointer);

        F.setName(Name + BodySuffix);
        F.setVisibility(GlobalValue::HiddenVisibility);

        auto KV = std::make_pair(std::move(Name), std::move(CallbackInfo));
        NewStubInfos.push_back(StubInfos.insert(StubInfos.begin(), KV));
      }

      auto H = addModule(std::move(SubM), MSI, LogicalModule, FallbackLookup);

      // Set the compile actions for this module:
      for (auto &KVPair : NewStubInfos) {
        std::string BodyName = Mangle(KVPair->first + BodySuffix,
                                      *M.getDataLayout());
        auto &CCInfo = KVPair->second;
        CCInfo.setCompileAction(
          [=](){
            return BaseLayer.findSymbolIn(H, BodyName, false).getAddress();
          });
      }

    }

    // Ok - we've processed all the partitioned modules. Now add the
    // stubs/globals module and set the update actions.
    auto StubsH =
      addModule(std::move(StubsModule), MSI, LogicalModule, FallbackLookup);

    for (auto &KVPair : StubInfos) {
      std::string AddrName = Mangle(KVPair.first + AddrSuffix,
                                    *M.getDataLayout());
      auto &CCInfo = KVPair.second;
      CCInfo.setUpdateAction(
        CompileCallbackMgr.getLocalFPUpdater(StubsH, AddrName));
    }
  }

  // Add the given Module to the base layer using a memory manager that will
  // perform the appropriate scoped lookup (i.e. will look first with in the
  // module from which it was extracted, then into the set to which that module
  // belonged, and finally externally).
  BaseLayerModuleSetHandleT addModule(
                               std::unique_ptr<Module> M,
                               ModuleSetInfo &MSI,
                               typename CODScopedLookup::LMHandle LogicalModule,
                               LookupFtor FallbackLookup) {

    // Add this module to the JIT with a memory manager that uses the
    // DylibLookup to resolve symbols.
    std::vector<std::unique_ptr<Module>> MSet;
    MSet.push_back(std::move(M));

    auto DylibLookup = MSI.Lookup;
    auto MM =
      createLookasideRTDyldMM<SectionMemoryManager>(
        [=](const std::string &Name) {
          if (auto Symbol = DylibLookup->findSymbol(LogicalModule, Name))
            return Symbol.getAddress();
          return FallbackLookup(Name);
        },
        [=](const std::string &Name) {
          return DylibLookup->findSymbol(LogicalModule, Name).getAddress();
        });

    BaseLayerModuleSetHandleT H =
      BaseLayer.addModuleSet(std::move(MSet), std::move(MM));
    // Add this module to the logical module lookup.
    DylibLookup->addToLogicalModule(LogicalModule, H);
    MSI.BaseLayerModuleSetHandles.push_back(H);

    return H;
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
  CompileCallbackMgrT CompileCallbackMgr;
  ModuleSetInfoListT ModuleSetInfos;
};

} // End namespace orc.
} // End namespace llvm.

#endif // LLVM_EXECUTIONENGINE_ORC_COMPILEONDEMANDLAYER_H
