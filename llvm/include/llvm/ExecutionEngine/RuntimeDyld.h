//===-- RuntimeDyld.h - Run-time dynamic linker for MC-JIT ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interface for the runtime dynamic linker facilities of the MC-JIT.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_RUNTIMEDYLD_H
#define LLVM_EXECUTIONENGINE_RUNTIMEDYLD_H

#include "JITSymbolFlags.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/Support/Memory.h"
#include <memory>

namespace llvm {

namespace object {
  class ObjectFile;
  template <typename T> class OwningBinary;
}

class RuntimeDyldImpl;
class RuntimeDyldCheckerImpl;
 
class RuntimeDyld {
  friend class RuntimeDyldCheckerImpl;

  RuntimeDyld(const RuntimeDyld &) = delete;
  void operator=(const RuntimeDyld &) = delete;

  // RuntimeDyldImpl is the actual class. RuntimeDyld is just the public
  // interface.
  std::unique_ptr<RuntimeDyldImpl> Dyld;
  RTDyldMemoryManager *MM;
  bool ProcessAllSections;
  RuntimeDyldCheckerImpl *Checker;
protected:
  // Change the address associated with a section when resolving relocations.
  // Any relocations already associated with the symbol will be re-resolved.
  void reassignSectionAddress(unsigned SectionID, uint64_t Addr);
public:

  /// \brief Information about a named symbol.
  class SymbolInfo : public JITSymbolBase {
  public:
    SymbolInfo(std::nullptr_t) : JITSymbolBase(JITSymbolFlags::None), Address(0) {}
    SymbolInfo(uint64_t Address, JITSymbolFlags Flags)
      : JITSymbolBase(Flags), Address(Address) {}
    explicit operator bool() const { return Address != 0; }
    uint64_t getAddress() const { return Address; }
  private:
    uint64_t Address;
  };

  /// \brief Information about the loaded object.
  class LoadedObjectInfo {
    friend class RuntimeDyldImpl;
  public:
    LoadedObjectInfo(RuntimeDyldImpl &RTDyld, unsigned BeginIdx,
                     unsigned EndIdx)
      : RTDyld(RTDyld), BeginIdx(BeginIdx), EndIdx(EndIdx) { }

    virtual ~LoadedObjectInfo() {}

    virtual object::OwningBinary<object::ObjectFile>
    getObjectForDebug(const object::ObjectFile &Obj) const = 0;

    uint64_t getSectionLoadAddress(StringRef Name) const;

  protected:
    virtual void anchor();

    RuntimeDyldImpl &RTDyld;
    unsigned BeginIdx, EndIdx;
  };

  RuntimeDyld(RTDyldMemoryManager *);
  ~RuntimeDyld();

  /// Add the referenced object file to the list of objects to be loaded and
  /// relocated.
  std::unique_ptr<LoadedObjectInfo> loadObject(const object::ObjectFile &O);

  /// Get the address of our local copy of the symbol. This may or may not
  /// be the address used for relocation (clients can copy the data around
  /// and resolve relocatons based on where they put it).
  void *getSymbolLocalAddress(StringRef Name) const;

  /// Get the target address and flags for the named symbol.
  /// This address is the one used for relocation.
  SymbolInfo getSymbol(StringRef Name) const;

  /// Resolve the relocations for all symbols we currently know about.
  void resolveRelocations();

  /// Map a section to its target address space value.
  /// Map the address of a JIT section as returned from the memory manager
  /// to the address in the target process as the running code will see it.
  /// This is the address which will be used for relocation resolution.
  void mapSectionAddress(const void *LocalAddress, uint64_t TargetAddress);

  /// Register any EH frame sections that have been loaded but not previously
  /// registered with the memory manager.  Note, RuntimeDyld is responsible
  /// for identifying the EH frame and calling the memory manager with the
  /// EH frame section data.  However, the memory manager itself will handle
  /// the actual target-specific EH frame registration.
  void registerEHFrames();

  void deregisterEHFrames();

  bool hasError();
  StringRef getErrorString();

  /// By default, only sections that are "required for execution" are passed to
  /// the RTDyldMemoryManager, and other sections are discarded. Passing 'true'
  /// to this method will cause RuntimeDyld to pass all sections to its
  /// memory manager regardless of whether they are "required to execute" in the
  /// usual sense. This is useful for inspecting metadata sections that may not
  /// contain relocations, E.g. Debug info, stackmaps.
  ///
  /// Must be called before the first object file is loaded.
  void setProcessAllSections(bool ProcessAllSections) {
    assert(!Dyld && "setProcessAllSections must be called before loadObject.");
    this->ProcessAllSections = ProcessAllSections;
  }
};

} // end namespace llvm

#endif
