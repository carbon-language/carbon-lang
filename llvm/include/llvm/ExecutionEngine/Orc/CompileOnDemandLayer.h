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
#include <list>

namespace llvm {

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
template <typename BaseLayerT> class CompileOnDemandLayer {
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
    uint64_t lookup(LMHandle LMH, const std::string &Name) {
      if (uint64_t Addr = lookupOnlyIn(LMH, Name))
        return Addr;

      for (auto I = Handles.begin(), E = Handles.end(); I != E; ++I)
        if (I != LMH)
          if (uint64_t Addr = lookupOnlyIn(I, Name))
            return Addr;

      return 0;
    }

  private:
    uint64_t lookupOnlyIn(LMHandle LMH, const std::string &Name) {
      for (auto H : *LMH)
        if (uint64_t Addr = BaseLayer.lookupSymbolAddressIn(H, Name, false))
          return Addr;
      return 0;
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

    // Persistent manglers - one per TU.
    std::vector<PersistentMangler> PersistentManglers;

    // Symbol resolution callback handlers - one per TU.
    std::vector<std::unique_ptr<JITResolveCallbackHandler>>
        JITResolveCallbackHandlers;

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

  /// @brief Convenience typedef for callback inserter.
  typedef std::function<void(Module&, JITResolveCallbackHandler&)>
    InsertCallbackAsmFtor;

  /// @brief Construct a compile-on-demand layer instance.
  CompileOnDemandLayer(BaseLayerT &BaseLayer,
                       InsertCallbackAsmFtor InsertCallbackAsm)
    : BaseLayer(BaseLayer), InsertCallbackAsm(InsertCallbackAsm) {}

  /// @brief Add a module to the compile-on-demand layer.
  template <typename ModuleSetT>
  ModuleSetHandleT addModuleSet(ModuleSetT Ms,
                                std::unique_ptr<RTDyldMemoryManager> MM) {

    const char *JITAddrSuffix = "$orc_addr";
    const char *JITImplSuffix = "$orc_impl";

    // Create a symbol lookup context and ModuleSetInfo for this module set.
    auto DylibLookup = std::make_shared<CODScopedLookup>(BaseLayer);
    ModuleSetHandleT H =
        ModuleSetInfos.insert(ModuleSetInfos.end(), ModuleSetInfo(DylibLookup));
    ModuleSetInfo &MSI = ModuleSetInfos.back();

    // Process each of the modules in this module set. All modules share the
    // same lookup context, but each will get its own TU lookup context.
    for (auto &M : Ms) {

      // Create a TU lookup context for this module.
      auto LMH = DylibLookup->createLogicalModule();
      MSI.LMHandles.push_back(LMH);

      // Create a persistent mangler for this module.
      MSI.PersistentManglers.emplace_back(*M->getDataLayout());

      // Make all calls to functions defined in this module indirect.
      JITIndirections Indirections =
          makeCallsDoubleIndirect(*M, [](const Function &) { return true; },
                                  JITImplSuffix, JITAddrSuffix);

      // Then carve up the module into a bunch of single-function modules.
      std::vector<std::unique_ptr<Module>> ExplodedModules =
          explode(*M, Indirections);

      // Add a resolve-callback handler for this module to look up symbol
      // addresses when requested via a callback.
      MSI.JITResolveCallbackHandlers.push_back(
          createCallbackHandlerFromJITIndirections(
              Indirections, MSI.PersistentManglers.back(),
              [=](StringRef S) { return DylibLookup->lookup(LMH, S); }));

      // Insert callback asm code into the first module.
      InsertCallbackAsm(*ExplodedModules[0],
                        *MSI.JITResolveCallbackHandlers.back());

      // Now we need to take each of the extracted Modules and add them to
      // base layer. Each Module will be added individually to make sure they
      // can be compiled separately, and each will get its own lookaside
      // memory manager with lookup functors that resolve symbols in sibling
      // modules first.OA
      for (auto &M : ExplodedModules) {
        std::vector<std::unique_ptr<Module>> MSet;
        MSet.push_back(std::move(M));

        BaseLayerModuleSetHandleT H = BaseLayer.addModuleSet(
            std::move(MSet),
            createLookasideRTDyldMM<SectionMemoryManager>(
                [=](const std::string &Name) {
                  if (uint64_t Addr = DylibLookup->lookup(LMH, Name))
                    return Addr;
                  return getSymbolAddress(Name, true);
                },
                [=](const std::string &Name) {
                  return DylibLookup->lookup(LMH, Name);
                }));
        DylibLookup->addToLogicalModule(LMH, H);
        MSI.BaseLayerModuleSetHandles.push_back(H);
      }

      initializeFuncAddrs(*MSI.JITResolveCallbackHandlers.back(), Indirections,
                          MSI.PersistentManglers.back(), [=](StringRef S) {
                            return DylibLookup->lookup(LMH, S);
                          });
    }

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

  /// @brief Get the address of a symbol provided by this layer, or some layer
  ///        below this one.
  uint64_t getSymbolAddress(const std::string &Name, bool ExportedSymbolsOnly) {
    return BaseLayer.getSymbolAddress(Name, ExportedSymbolsOnly);
  }

  /// @brief Get the address of a symbol provided by this layer, or some layer
  ///        below this one.
  uint64_t lookupSymbolAddressIn(ModuleSetHandleT H, const std::string &Name,
                                 bool ExportedSymbolsOnly) {
    BaseLayerModuleSetHandleListT &BaseLayerHandles = H->second;
    for (auto &BH : BaseLayerHandles) {
      if (uint64_t Addr =
            BaseLayer.lookupSymbolAddressIn(BH, Name, ExportedSymbolsOnly))
        return Addr;
    }
    return 0;
  }

private:
  BaseLayerT &BaseLayer;
  InsertCallbackAsmFtor InsertCallbackAsm;
  ModuleSetInfoListT ModuleSetInfos;
};
}

#endif // LLVM_EXECUTIONENGINE_ORC_COMPILEONDEMANDLAYER_H
