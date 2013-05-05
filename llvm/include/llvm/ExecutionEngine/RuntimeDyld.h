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

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/ObjectBuffer.h"
#include "llvm/Support/Memory.h"

namespace llvm {

class RuntimeDyldImpl;
class ObjectImage;

// RuntimeDyld clients often want to handle the memory management of
// what gets placed where. For JIT clients, this is the subset of
// JITMemoryManager required for dynamic loading of binaries.
//
// FIXME: As the RuntimeDyld fills out, additional routines will be needed
//        for the varying types of objects to be allocated.
class RTDyldMemoryManager {
  RTDyldMemoryManager(const RTDyldMemoryManager&) LLVM_DELETED_FUNCTION;
  void operator=(const RTDyldMemoryManager&) LLVM_DELETED_FUNCTION;
public:
  RTDyldMemoryManager() {}
  virtual ~RTDyldMemoryManager();

  /// Allocate a memory block of (at least) the given size suitable for
  /// executable code. The SectionID is a unique identifier assigned by the JIT
  /// engine, and optionally recorded by the memory manager to access a loaded
  /// section.
  virtual uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                                       unsigned SectionID) = 0;

  /// Allocate a memory block of (at least) the given size suitable for data.
  /// The SectionID is a unique identifier assigned by the JIT engine, and
  /// optionally recorded by the memory manager to access a loaded section.
  virtual uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                                       unsigned SectionID, bool IsReadOnly) = 0;

  /// This method returns the address of the specified function. As such it is
  /// only useful for resolving library symbols, not code generated symbols.
  ///
  /// If AbortOnFailure is false and no function with the given name is
  /// found, this function returns a null pointer. Otherwise, it prints a
  /// message to stderr and aborts.
  virtual void *getPointerToNamedFunction(const std::string &Name,
                                          bool AbortOnFailure = true) = 0;

  /// This method is called when object loading is complete and section page
  /// permissions can be applied.  It is up to the memory manager implementation
  /// to decide whether or not to act on this method.  The memory manager will
  /// typically allocate all sections as read-write and then apply specific
  /// permissions when this method is called.
  ///
  /// Returns true if an error occurred, false otherwise.
  virtual bool applyPermissions(std::string *ErrMsg = 0) = 0;

  /// Register the EH frames with the runtime so that c++ exceptions work. The
  /// default implementation does nothing. Look at SectionMemoryManager for one
  /// that uses __register_frame.
  virtual void registerEHFrames(StringRef SectionData);
};

class RuntimeDyld {
  RuntimeDyld(const RuntimeDyld &) LLVM_DELETED_FUNCTION;
  void operator=(const RuntimeDyld &) LLVM_DELETED_FUNCTION;

  // RuntimeDyldImpl is the actual class. RuntimeDyld is just the public
  // interface.
  RuntimeDyldImpl *Dyld;
  RTDyldMemoryManager *MM;
protected:
  // Change the address associated with a section when resolving relocations.
  // Any relocations already associated with the symbol will be re-resolved.
  void reassignSectionAddress(unsigned SectionID, uint64_t Addr);
public:
  RuntimeDyld(RTDyldMemoryManager *);
  ~RuntimeDyld();

  /// Prepare the object contained in the input buffer for execution.
  /// Ownership of the input buffer is transferred to the ObjectImage
  /// instance returned from this function if successful. In the case of load
  /// failure, the input buffer will be deleted.
  ObjectImage *loadObject(ObjectBuffer *InputBuffer);

  /// Get the address of our local copy of the symbol. This may or may not
  /// be the address used for relocation (clients can copy the data around
  /// and resolve relocatons based on where they put it).
  void *getSymbolAddress(StringRef Name);

  /// Get the address of the target copy of the symbol. This is the address
  /// used for relocation.
  uint64_t getSymbolLoadAddress(StringRef Name);

  /// Resolve the relocations for all symbols we currently know about.
  void resolveRelocations();

  /// Map a section to its target address space value.
  /// Map the address of a JIT section as returned from the memory manager
  /// to the address in the target process as the running code will see it.
  /// This is the address which will be used for relocation resolution.
  void mapSectionAddress(const void *LocalAddress, uint64_t TargetAddress);

  StringRef getErrorString();

  StringRef getEHFrameSection();
};

} // end namespace llvm

#endif
