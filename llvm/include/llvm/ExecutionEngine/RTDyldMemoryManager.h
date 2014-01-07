//===-- RTDyldMemoryManager.cpp - Memory manager for MC-JIT -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interface of the runtime dynamic memory manager base class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_RT_DYLD_MEMORY_MANAGER_H
#define LLVM_EXECUTIONENGINE_RT_DYLD_MEMORY_MANAGER_H

#include "llvm-c/ExecutionEngine.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/Memory.h"

namespace llvm {

class ExecutionEngine;
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
  virtual uint8_t *allocateCodeSection(
    uintptr_t Size, unsigned Alignment, unsigned SectionID,
    StringRef SectionName) = 0;

  /// Allocate a memory block of (at least) the given size suitable for data.
  /// The SectionID is a unique identifier assigned by the JIT engine, and
  /// optionally recorded by the memory manager to access a loaded section.
  virtual uint8_t *allocateDataSection(
    uintptr_t Size, unsigned Alignment, unsigned SectionID,
    StringRef SectionName, bool IsReadOnly) = 0;

  /// Register the EH frames with the runtime so that c++ exceptions work.
  ///
  /// \p Addr parameter provides the local address of the EH frame section
  /// data, while \p LoadAddr provides the address of the data in the target
  /// address space.  If the section has not been remapped (which will usually
  /// be the case for local execution) these two values will be the same.
  virtual void registerEHFrames(uint8_t *Addr, uint64_t LoadAddr, size_t Size);

  virtual void deregisterEHFrames(uint8_t *Addr, uint64_t LoadAddr, size_t Size);

  /// This method returns the address of the specified function or variable.
  /// It is used to resolve symbols during module linking.
  virtual uint64_t getSymbolAddress(const std::string &Name);

  /// This method returns the address of the specified function. As such it is
  /// only useful for resolving library symbols, not code generated symbols.
  ///
  /// If \p AbortOnFailure is false and no function with the given name is
  /// found, this function returns a null pointer. Otherwise, it prints a
  /// message to stderr and aborts.
  ///
  /// This function is deprecated for memory managers to be used with
  /// MCJIT or RuntimeDyld.  Use getSymbolAddress instead.
  virtual void *getPointerToNamedFunction(const std::string &Name,
                                          bool AbortOnFailure = true);

  /// This method is called after an object has been loaded into memory but
  /// before relocations are applied to the loaded sections.  The object load
  /// may have been initiated by MCJIT to resolve an external symbol for another
  /// object that is being finalized.  In that case, the object about which
  /// the memory manager is being notified will be finalized immediately after
  /// the memory manager returns from this call.
  ///
  /// Memory managers which are preparing code for execution in an external
  /// address space can use this call to remap the section addresses for the
  /// newly loaded object.
  virtual void notifyObjectLoaded(ExecutionEngine *EE,
                                  const ObjectImage *) {}

  /// This method is called when object loading is complete and section page
  /// permissions can be applied.  It is up to the memory manager implementation
  /// to decide whether or not to act on this method.  The memory manager will
  /// typically allocate all sections as read-write and then apply specific
  /// permissions when this method is called.  Code sections cannot be executed
  /// until this function has been called.  In addition, any cache coherency
  /// operations needed to reliably use the memory are also performed.
  ///
  /// Returns true if an error occurred, false otherwise.
  virtual bool finalizeMemory(std::string *ErrMsg = 0) = 0;
};

// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(
    RTDyldMemoryManager, LLVMMCJITMemoryManagerRef)

} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_RT_DYLD_MEMORY_MANAGER_H
