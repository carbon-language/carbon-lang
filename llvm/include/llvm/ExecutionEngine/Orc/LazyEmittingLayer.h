//===- LazyEmittingLayer.h - Lazily emit IR to lower JIT layers -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Contains the definition for a lazy-emitting layer for the JIT.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_LAZYEMITTINGLAYER_H
#define LLVM_EXECUTIONENGINE_ORC_LAZYEMITTINGLAYER_H

#include "LookasideRTDyldMM.h"
#include "llvm/IR/Mangler.h"
#include <list>

namespace llvm {

/// @brief Lazy-emitting IR layer.
///
///   This layer accepts sets of LLVM IR Modules (via addModuleSet), but does
/// not immediately emit them the layer below. Instead, emissing to the base
/// layer is deferred until some symbol in the module set is requested via
/// getSymbolAddress.
template <typename BaseLayerT> class LazyEmittingLayer {
public:
  typedef typename BaseLayerT::ModuleSetHandleT BaseLayerHandleT;

private:
  class EmissionDeferredSet {
  public:
    EmissionDeferredSet() : EmitState(NotEmitted) {}
    virtual ~EmissionDeferredSet() {}

    uint64_t Search(StringRef Name, bool ExportedSymbolsOnly, BaseLayerT &B) {
      switch (EmitState) {
        case NotEmitted:
          if (Provides(Name, ExportedSymbolsOnly)) {
            EmitState = Emitting;
            Handle = Emit(B);
            EmitState = Emitted;
          } else
            return 0;
          break;
        case Emitting: 
          // The module has been added to the base layer but we haven't gotten a
          // handle back yet so we can't use lookupSymbolAddressIn. Just return
          // '0' here - LazyEmittingLayer::getSymbolAddress will do a global
          // search in the base layer when it doesn't find the symbol here, so
          // we'll find it in the end.
          return 0;
        case Emitted:
          // Nothing to do. Go ahead and search the base layer.
          break;
      }

      return B.lookupSymbolAddressIn(Handle, Name, ExportedSymbolsOnly);
    }

    void RemoveModulesFromBaseLayer(BaseLayerT &BaseLayer) {
      if (EmitState != NotEmitted)
        BaseLayer.removeModuleSet(Handle);
    }

    template <typename ModuleSetT>
    static std::unique_ptr<EmissionDeferredSet>
    create(BaseLayerT &B, ModuleSetT Ms,
           std::unique_ptr<RTDyldMemoryManager> MM);

  protected:
    virtual bool Provides(StringRef Name, bool ExportedSymbolsOnly) const = 0;
    virtual BaseLayerHandleT Emit(BaseLayerT &BaseLayer) = 0;

  private:
    enum { NotEmitted, Emitting, Emitted } EmitState;
    BaseLayerHandleT Handle;
  };

  template <typename ModuleSetT>
  class EmissionDeferredSetImpl : public EmissionDeferredSet {
  public:
    EmissionDeferredSetImpl(ModuleSetT Ms,
                            std::unique_ptr<RTDyldMemoryManager> MM)
        : Ms(std::move(Ms)), MM(std::move(MM)) {}

  protected:
    BaseLayerHandleT Emit(BaseLayerT &BaseLayer) override {
      // We don't need the mangled names set any more: Once we've emitted this
      // to the base layer we'll just look for symbols there.
      MangledNames.reset();
      return BaseLayer.addModuleSet(std::move(Ms), std::move(MM));
    }

    bool Provides(StringRef Name, bool ExportedSymbolsOnly) const override {
      // FIXME: We could clean all this up if we had a way to reliably demangle
      //        names: We could just demangle name and search, rather than
      //        mangling everything else.

      // If we have already built the mangled name set then just search it.
      if (MangledNames) {
        auto VI = MangledNames->find(Name);
        if (VI == MangledNames->end())
          return false;
        return !ExportedSymbolsOnly || VI->second;
      }

      // If we haven't built the mangled name set yet, try to build it. As an
      // optimization this will leave MangledNames set to nullptr if we find
      // Name in the process of building the set.
      buildMangledNames(Name, ExportedSymbolsOnly);
      if (!MangledNames)
        return true;
      return false;
    }

  private:
    // If the mangled name of the given GlobalValue matches the given search
    // name (and its visibility conforms to the ExportedSymbolsOnly flag) then
    // just return 'true'. Otherwise, add the mangled name to the Names map and
    // return 'false'.
    bool addGlobalValue(StringMap<bool> &Names, const GlobalValue &GV,
                        const Mangler &Mang, StringRef SearchName,
                        bool ExportedSymbolsOnly) const {
      // Modules don't "provide" decls or common symbols.
      if (GV.isDeclaration() || GV.hasCommonLinkage())
        return false;

      // Mangle the GV name.
      std::string MangledName;
      {
        raw_string_ostream MangledNameStream(MangledName);
        Mang.getNameWithPrefix(MangledNameStream, &GV, false);
      }

      // Check whether this is the name we were searching for, and if it is then
      // bail out early.
      if (MangledName == SearchName)
        if (!ExportedSymbolsOnly || GV.hasDefaultVisibility())
          return true;

      // Otherwise add this to the map for later.
      Names[MangledName] = GV.hasDefaultVisibility();
      return false;
    }

    // Build the MangledNames map. Bails out early (with MangledNames left set
    // to nullptr) if the given SearchName is found while building the map.
    void buildMangledNames(StringRef SearchName,
                           bool ExportedSymbolsOnly) const {
      assert(!MangledNames && "Mangled names map already exists?");

      auto Names = llvm::make_unique<StringMap<bool>>();

      for (const auto &M : Ms) {
        Mangler Mang(M->getDataLayout());

        for (const auto &GV : M->globals())
          if (addGlobalValue(*Names, GV, Mang, SearchName, ExportedSymbolsOnly))
            return;

        for (const auto &F : *M)
          if (addGlobalValue(*Names, F, Mang, SearchName, ExportedSymbolsOnly))
            return;
      }

      MangledNames = std::move(Names);
    }

    ModuleSetT Ms;
    std::unique_ptr<RTDyldMemoryManager> MM;
    mutable std::unique_ptr<StringMap<bool>> MangledNames;
  };

  typedef std::list<std::unique_ptr<EmissionDeferredSet>> ModuleSetListT;

  BaseLayerT &BaseLayer;
  ModuleSetListT ModuleSetList;

public:
  /// @brief Handle to a set of loaded modules.
  typedef typename ModuleSetListT::iterator ModuleSetHandleT;

  /// @brief Construct a lazy emitting layer.
  LazyEmittingLayer(BaseLayerT &BaseLayer) : BaseLayer(BaseLayer) {}

  /// @brief Add the given set of modules to the lazy emitting layer.
  ///
  ///   This method stores the set of modules in a side table, rather than
  /// immediately emitting them to the next layer of the JIT. When the address
  /// of a symbol provided by this set is requested (via getSymbolAddress) it
  /// triggers the emission of this set to the layer below (along with the given
  /// memory manager instance), and returns the address of the requested symbol.
  template <typename ModuleSetT>
  ModuleSetHandleT addModuleSet(ModuleSetT Ms,
                                std::unique_ptr<RTDyldMemoryManager> MM) {
    return ModuleSetList.insert(
        ModuleSetList.end(),
        EmissionDeferredSet::create(BaseLayer, std::move(Ms), std::move(MM)));
  }

  /// @brief Remove the module set represented by the given handle.
  ///
  ///   This method will free the memory associated with the given module set,
  /// both in this layer, and the base layer.
  void removeModuleSet(ModuleSetHandleT H) {
    (*H)->RemoveModulesFromBaseLayer(BaseLayer);
    ModuleSetList.erase(H);
  }

  /// @brief Get the address of a symbol provided by this layer, or some layer
  ///        below this one.
  ///
  ///   When called for a symbol that has been added to this layer (via
  /// addModuleSet) but not yet emitted, this will trigger the emission of the
  /// module set containing the definiton of the symbol.
  uint64_t getSymbolAddress(const std::string &Name, bool ExportedSymbolsOnly) {
    // Look up symbol among existing definitions.
    if (uint64_t Addr = BaseLayer.getSymbolAddress(Name, ExportedSymbolsOnly))
      return Addr;

    // If not found then search the deferred sets. The call to 'Search' will
    // cause the set to be emitted to the next layer if it provides a definition
    // of 'Name'.
    for (auto &DeferredSet : ModuleSetList)
      if (uint64_t Addr =
              DeferredSet->Search(Name, ExportedSymbolsOnly, BaseLayer))
        return Addr;

    // If no definition found anywhere return 0.
    return 0;
  }

  /// @brief Get the address of the given symbol in the context of the set of
  ///        compiled modules represented by the handle H. This call is
  ///        forwarded to the base layer's implementation.
  uint64_t lookupSymbolAddressIn(ModuleSetHandleT H, const std::string &Name,
                                 bool ExportedSymbolsOnly) {
    return (*H)->Search(Name, ExportedSymbolsOnly, BaseLayer);
  }
};

template <typename BaseLayerT>
template <typename ModuleSetT>
std::unique_ptr<typename LazyEmittingLayer<BaseLayerT>::EmissionDeferredSet>
LazyEmittingLayer<BaseLayerT>::EmissionDeferredSet::create(
    BaseLayerT &B, ModuleSetT Ms, std::unique_ptr<RTDyldMemoryManager> MM) {
  return llvm::make_unique<EmissionDeferredSetImpl<ModuleSetT>>(std::move(Ms),
                                                                std::move(MM));
}
}

#endif // LLVM_EXECUTIONENGINE_ORC_LAZYEMITTINGLAYER_H
