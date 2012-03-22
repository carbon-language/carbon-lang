//===-- RuntimeDyldImpl.h - Run-time dynamic linker for MC-JIT --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interface for the implementations of runtime dynamic linker facilities.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_RUNTIME_DYLD_IMPL_H
#define LLVM_RUNTIME_DYLD_IMPL_H

#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/system_error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace llvm {
class RuntimeDyldImpl {
protected:
  unsigned CPUType;
  unsigned CPUSubtype;

  // The MemoryManager to load objects into.
  RTDyldMemoryManager *MemMgr;

  // For each section, we have a MemoryBlock of it's data.
  // Indexed by SectionID.
  SmallVector<sys::MemoryBlock, 32> Sections;
  // For each section, the address it will be considered to live at for
  // relocations. The same as the pointer to the above memory block for hosted
  // JITs. Indexed by SectionID.
  SmallVector<uint64_t, 32> SectionLoadAddress;

  // Keep a map of starting local address to the SectionID which references it.
  // Lookup function for when we assign virtual addresses.
  DenseMap<void *, unsigned> SectionLocalMemToID;

  // Master symbol table. As modules are loaded and external symbols are
  // resolved, their addresses are stored here as a SectionID/Offset pair.
  typedef std::pair<unsigned, uint64_t> SymbolLoc;
  StringMap<SymbolLoc> SymbolTable;

  bool HasError;
  std::string ErrorStr;

  // Set the error state and record an error string.
  bool Error(const Twine &Msg) {
    ErrorStr = Msg.str();
    HasError = true;
    return true;
  }

  uint8_t *getSectionAddress(unsigned SectionID) {
    return (uint8_t*)Sections[SectionID].base();
  }
  void extractFunction(StringRef Name, uint8_t *StartAddress,
                       uint8_t *EndAddress);

public:
  RuntimeDyldImpl(RTDyldMemoryManager *mm) : MemMgr(mm), HasError(false) {}

  virtual ~RuntimeDyldImpl();

  virtual bool loadObject(MemoryBuffer *InputBuffer) = 0;

  void *getSymbolAddress(StringRef Name) {
    // FIXME: Just look up as a function for now. Overly simple of course.
    // Work in progress.
    if (SymbolTable.find(Name) == SymbolTable.end())
      return 0;
    SymbolLoc Loc = SymbolTable.lookup(Name);
    return getSectionAddress(Loc.first) + Loc.second;
  }

  virtual void resolveRelocations();

  virtual void reassignSectionAddress(unsigned SectionID, uint64_t Addr) = 0;

  void mapSectionAddress(void *LocalAddress, uint64_t TargetAddress);

  // Is the linker in an error state?
  bool hasError() { return HasError; }

  // Mark the error condition as handled and continue.
  void clearError() { HasError = false; }

  // Get the error message.
  StringRef getErrorString() { return ErrorStr; }

  virtual bool isCompatibleFormat(const MemoryBuffer *InputBuffer) const = 0;
};

} // end namespace llvm


#endif
