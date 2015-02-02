//===- ObjectLinkingLayer.h - Add object files to a JIT process -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Contains the definition for the object layer of the JIT.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_OBJECTLINKINGLAYER_H
#define LLVM_EXECUTIONENGINE_ORC_OBJECTLINKINGLAYER_H

#include "LookasideRTDyldMM.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include <list>
#include <memory>

namespace llvm {

class ObjectLinkingLayerBase {
protected:
  /// @brief Holds a set of objects to be allocated/linked as a unit in the JIT.
  ///
  /// An instance of this class will be created for each set of objects added
  /// via JITObjectLayer::addObjectSet. Deleting the instance (via
  /// removeObjectSet) frees its memory, removing all symbol definitions that
  /// had been provided by this instance. Higher level layers are responsible
  /// for taking any action required to handle the missing symbols.
  class LinkedObjectSet {
    LinkedObjectSet(const LinkedObjectSet&) LLVM_DELETED_FUNCTION;
    void operator=(const LinkedObjectSet&) LLVM_DELETED_FUNCTION;
  public:
    LinkedObjectSet(std::unique_ptr<RTDyldMemoryManager> MM)
        : MM(std::move(MM)), RTDyld(llvm::make_unique<RuntimeDyld>(&*this->MM)),
          State(Raw) {}

    // MSVC 2012 cannot infer a move constructor, so write it out longhand.
    LinkedObjectSet(LinkedObjectSet &&O)
        : MM(std::move(O.MM)), RTDyld(std::move(O.RTDyld)), State(O.State) {}

    std::unique_ptr<RuntimeDyld::LoadedObjectInfo>
    addObject(const object::ObjectFile &Obj) {
      return RTDyld->loadObject(Obj);
    }

    uint64_t getSymbolAddress(StringRef Name, bool ExportedSymbolsOnly) {
      if (ExportedSymbolsOnly)
        return RTDyld->getExportedSymbolLoadAddress(Name);
      return RTDyld->getSymbolLoadAddress(Name);
    }

    bool NeedsFinalization() const { return (State == Raw); }

    void Finalize() {
      State = Finalizing;
      RTDyld->resolveRelocations();
      RTDyld->registerEHFrames();
      MM->finalizeMemory();
      OwnedBuffers.clear();
      State = Finalized;
    }

    void mapSectionAddress(const void *LocalAddress, uint64_t TargetAddress) {
      assert((State != Finalized) &&
             "Attempting to remap sections for finalized objects.");
      RTDyld->mapSectionAddress(LocalAddress, TargetAddress);
    }

    void takeOwnershipOfBuffer(std::unique_ptr<MemoryBuffer> B) {
      OwnedBuffers.push_back(std::move(B));
    }

  private:
    std::unique_ptr<RTDyldMemoryManager> MM;
    std::unique_ptr<RuntimeDyld> RTDyld;
    enum { Raw, Finalizing, Finalized } State;

    // FIXME: This ownership hack only exists because RuntimeDyldELF still
    //        wants to be able to inspect the original object when resolving
    //        relocations. As soon as that can be fixed this should be removed.
    std::vector<std::unique_ptr<MemoryBuffer>> OwnedBuffers;
  };

  typedef std::list<LinkedObjectSet> LinkedObjectSetListT;

public:
  /// @brief Handle to a set of loaded objects.
  typedef LinkedObjectSetListT::iterator ObjSetHandleT;

  // Ownership hack.
  // FIXME: Remove this as soon as RuntimeDyldELF can apply relocations without
  //        referencing the original object.
  template <typename OwningMBSet>
  void takeOwnershipOfBuffers(ObjSetHandleT H, OwningMBSet MBs) {
    for (auto &MB : MBs)
      H->takeOwnershipOfBuffer(std::move(MB));
  }

};

/// @brief Default (no-op) action to perform when loading objects.
class DoNothingOnNotifyLoaded {
public:
  template <typename ObjSetT, typename LoadResult>
  void operator()(ObjectLinkingLayerBase::ObjSetHandleT, const ObjSetT &,
                  const LoadResult &) {}
};

/// @brief Bare bones object linking layer.
///
///   This class is intended to be used as the base layer for a JIT. It allows
/// object files to be loaded into memory, linked, and the addresses of their
/// symbols queried. All objects added to this layer can see each other's
/// symbols.
template <typename NotifyLoadedFtor = DoNothingOnNotifyLoaded>
class ObjectLinkingLayer : public ObjectLinkingLayerBase {
public:

  /// @brief LoadedObjectInfo list. Contains a list of owning pointers to
  ///        RuntimeDyld::LoadedObjectInfo instances.
  typedef std::vector<std::unique_ptr<RuntimeDyld::LoadedObjectInfo>>
      LoadedObjInfoList;

  /// @brief Functor to create RTDyldMemoryManager instances.
  typedef std::function<std::unique_ptr<RTDyldMemoryManager>()> CreateRTDyldMMFtor;

  /// @brief Functor for receiving finalization notifications.
  typedef std::function<void(ObjSetHandleT)> NotifyFinalizedFtor;

  /// @brief Construct an ObjectLinkingLayer with the given NotifyLoaded,
  ///        NotifyFinalized and CreateMemoryManager functors.
  ObjectLinkingLayer(
      CreateRTDyldMMFtor CreateMemoryManager = CreateRTDyldMMFtor(),
      NotifyLoadedFtor NotifyLoaded = NotifyLoadedFtor(),
      NotifyFinalizedFtor NotifyFinalized = NotifyFinalizedFtor())
      : NotifyLoaded(std::move(NotifyLoaded)),
        NotifyFinalized(std::move(NotifyFinalized)),
        CreateMemoryManager(std::move(CreateMemoryManager)) {}

  /// @brief Add a set of objects (or archives) that will be treated as a unit
  ///        for the purposes of symbol lookup and memory management.
  ///
  /// @return A pair containing (1) A handle that can be used to free the memory
  ///         allocated for the objects, and (2) a LoadedObjInfoList containing
  ///         one LoadedObjInfo instance for each object at the corresponding
  ///         index in the Objects list.
  ///
  ///   This version of this method allows the client to pass in an
  /// RTDyldMemoryManager instance that will be used to allocate memory and look
  /// up external symbol addresses for the given objects.
  template <typename ObjSetT>
  ObjSetHandleT addObjectSet(const ObjSetT &Objects,
                             std::unique_ptr<RTDyldMemoryManager> MM) {

    if (!MM) {
      assert(CreateMemoryManager &&
             "No memory manager or memory manager creator provided.");
      MM = CreateMemoryManager();
    }

    ObjSetHandleT Handle = LinkedObjSetList.insert(
        LinkedObjSetList.end(), LinkedObjectSet(std::move(MM)));
    LinkedObjectSet &LOS = *Handle;
    LoadedObjInfoList LoadedObjInfos;

    for (auto &Obj : Objects)
      LoadedObjInfos.push_back(LOS.addObject(*Obj));

    NotifyLoaded(Handle, Objects, LoadedObjInfos);

    return Handle;
  }

  /// @brief Map section addresses for the objects associated with the handle H.
  void mapSectionAddress(ObjSetHandleT H, const void *LocalAddress,
                         uint64_t TargetAddress) {
    H->mapSectionAddress(LocalAddress, TargetAddress);
  }

  /// @brief Remove the set of objects associated with handle H.
  ///
  ///   All memory allocated for the objects will be freed, and the sections and
  /// symbols they provided will no longer be available. No attempt is made to
  /// re-emit the missing symbols, and any use of these symbols (directly or
  /// indirectly) will result in undefined behavior. If dependence tracking is
  /// required to detect or resolve such issues it should be added at a higher
  /// layer.
  void removeObjectSet(ObjSetHandleT H) {
    // How do we invalidate the symbols in H?
    LinkedObjSetList.erase(H);
  }

  /// @brief Get the address of a loaded symbol.
  ///
  /// @return The address in the target process's address space of the named
  ///         symbol. Null if no such symbol is known.
  ///
  ///   This method will trigger the finalization of the linked object set
  /// containing the definition of the given symbol, if it is found.
  uint64_t getSymbolAddress(StringRef Name, bool ExportedSymbolsOnly) {
    for (auto I = LinkedObjSetList.begin(), E = LinkedObjSetList.end(); I != E;
         ++I)
      if (uint64_t Addr = lookupSymbolAddressIn(I, Name, ExportedSymbolsOnly))
        return Addr;

    return 0;
  }

  /// @brief Search for a given symbol in the context of the set of loaded
  ///        objects represented by the handle H.
  ///
  /// @return The address in the target process's address space of the named
  ///         symbol. Null if the given object set does not contain a definition
  ///         of this symbol.
  ///
  ///   This method will trigger the finalization of the linked object set
  /// represented by the handle H if that set contains the requested symbol.
  uint64_t lookupSymbolAddressIn(ObjSetHandleT H, StringRef Name,
                                 bool ExportedSymbolsOnly) {
    if (uint64_t Addr = H->getSymbolAddress(Name, ExportedSymbolsOnly)) {
      if (H->NeedsFinalization()) {
        H->Finalize();
        if (NotifyFinalized)
          NotifyFinalized(H);
      }
      return Addr;
    }
    return 0;
  }

private:
  LinkedObjectSetListT LinkedObjSetList;
  NotifyLoadedFtor NotifyLoaded;
  NotifyFinalizedFtor NotifyFinalized;
  CreateRTDyldMMFtor CreateMemoryManager;
};

} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_OBJECTLINKINGLAYER_H
