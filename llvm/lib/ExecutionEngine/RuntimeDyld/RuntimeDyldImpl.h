//===-- RuntimeDyldImpl.h - Run-time dynamic linker for MC-JIT ------*- C++ -*-===//
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
#include "llvm/Object/MachOObject.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/system_error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace llvm::object;

namespace llvm {
class RuntimeDyldImpl {
protected:
  unsigned CPUType;
  unsigned CPUSubtype;

  // The MemoryManager to load objects into.
  RTDyldMemoryManager *MemMgr;

  // FIXME: This all assumes we're dealing with external symbols for anything
  //        explicitly referenced. I.e., we can index by name and things
  //        will work out. In practice, this may not be the case, so we
  //        should find a way to effectively generalize.

  // For each function, we have a MemoryBlock of it's instruction data.
  StringMap<sys::MemoryBlock> Functions;

  // Master symbol table. As modules are loaded and external symbols are
  // resolved, their addresses are stored here.
  StringMap<uint8_t*> SymbolTable;

  bool HasError;
  std::string ErrorStr;

  // Set the error state and record an error string.
  bool Error(const Twine &Msg) {
    ErrorStr = Msg.str();
    HasError = true;
    return true;
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
    return SymbolTable.lookup(Name);
  }

  void resolveRelocations();

  virtual void reassignSymbolAddress(StringRef Name, uint8_t *Addr) = 0;

  // Is the linker in an error state?
  bool hasError() { return HasError; }

  // Mark the error condition as handled and continue.
  void clearError() { HasError = false; }

  // Get the error message.
  StringRef getErrorString() { return ErrorStr; }

  virtual bool isCompatibleFormat(const MemoryBuffer *InputBuffer) const = 0;
};


class RuntimeDyldMachO : public RuntimeDyldImpl {

  // For each symbol, keep a list of relocations based on it. Anytime
  // its address is reassigned (the JIT re-compiled the function, e.g.),
  // the relocations get re-resolved.
  struct RelocationEntry {
    std::string Target;     // Object this relocation is contained in.
    uint64_t    Offset;     // Offset into the object for the relocation.
    uint32_t    Data;       // Second word of the raw macho relocation entry.
    int64_t     Addend;     // Addend encoded in the instruction itself, if any.
    bool        isResolved; // Has this relocation been resolved previously?

    RelocationEntry(StringRef t, uint64_t offset, uint32_t data, int64_t addend)
      : Target(t), Offset(offset), Data(data), Addend(addend),
        isResolved(false) {}
  };
  typedef SmallVector<RelocationEntry, 4> RelocationList;
  StringMap<RelocationList> Relocations;

  // FIXME: Also keep a map of all the relocations contained in an object. Use
  // this to dynamically answer whether all of the relocations in it have
  // been resolved or not.

  bool resolveRelocation(uint8_t *Address, uint8_t *Value, bool isPCRel,
                         unsigned Type, unsigned Size);
  bool resolveX86_64Relocation(uintptr_t Address, uintptr_t Value, bool isPCRel,
                               unsigned Type, unsigned Size);
  bool resolveARMRelocation(uintptr_t Address, uintptr_t Value, bool isPCRel,
                            unsigned Type, unsigned Size);

  bool loadSegment32(const MachOObject *Obj,
                     const MachOObject::LoadCommandInfo *SegmentLCI,
                     const InMemoryStruct<macho::SymtabLoadCommand> &SymtabLC);
  bool loadSegment64(const MachOObject *Obj,
                     const MachOObject::LoadCommandInfo *SegmentLCI,
                     const InMemoryStruct<macho::SymtabLoadCommand> &SymtabLC);

public:
  RuntimeDyldMachO(RTDyldMemoryManager *mm) : RuntimeDyldImpl(mm) {}

  bool loadObject(MemoryBuffer *InputBuffer);

  void reassignSymbolAddress(StringRef Name, uint8_t *Addr);

  static bool isKnownFormat(const MemoryBuffer *InputBuffer);

  bool isCompatibleFormat(const MemoryBuffer *InputBuffer) const {
    return isKnownFormat(InputBuffer);
  };
};

} // end namespace llvm


#endif
