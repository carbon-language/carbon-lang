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
#include "llvm/Object/MachOObject.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IndexedMap.h"
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

class RuntimeDyldELF : public RuntimeDyldImpl {
    // For each symbol, keep a list of relocations based on it. Anytime
    // its address is reassigned (the JIT re-compiled the function, e.g.),
    // the relocations get re-resolved.
    struct RelocationEntry {
      // Function or section this relocation is contained in.
      std::string Target;
      // Offset into the target function or section for the relocation.
      uint32_t    Offset;
      // Relocation type
      uint32_t    Type;
      // Addend encoded in the instruction itself, if any.
      int32_t     Addend;
      // Has the relocation been recalcuated as an offset within a function?
      bool        IsFunctionRelative;
      // Has this relocation been resolved previously?
      bool        isResolved;

      RelocationEntry(StringRef t,
                      uint32_t offset,
                      uint32_t type,
                      int32_t addend,
                      bool isFunctionRelative)
        : Target(t)
        , Offset(offset)
        , Type(type)
        , Addend(addend)
        , IsFunctionRelative(isFunctionRelative)
        , isResolved(false) { }
    };
    typedef SmallVector<RelocationEntry, 4> RelocationList;
    StringMap<RelocationList> Relocations;
    unsigned Arch;

    void resolveRelocations();

    void resolveX86_64Relocation(StringRef Name,
                                 uint8_t *Addr,
                                 const RelocationEntry &RE);

    void resolveX86Relocation(StringRef Name,
                              uint8_t *Addr,
                              const RelocationEntry &RE);

    void resolveArmRelocation(StringRef Name,
                              uint8_t *Addr,
                              const RelocationEntry &RE);

    void resolveRelocation(StringRef Name,
                           uint8_t *Addr,
                           const RelocationEntry &RE);

public:
  RuntimeDyldELF(RTDyldMemoryManager *mm) : RuntimeDyldImpl(mm) {}

  bool loadObject(MemoryBuffer *InputBuffer);

  void reassignSymbolAddress(StringRef Name, uint8_t *Addr);
  void reassignSectionAddress(unsigned SectionID, uint64_t Addr);

  bool isCompatibleFormat(const MemoryBuffer *InputBuffer) const;
};


class RuntimeDyldMachO : public RuntimeDyldImpl {

  // For each symbol, keep a list of relocations based on it. Anytime
  // its address is reassigned (the JIT re-compiled the function, e.g.),
  // the relocations get re-resolved.
  // The symbol (or section) the relocation is sourced from is the Key
  // in the relocation list where it's stored.
  struct RelocationEntry {
    unsigned    SectionID;  // Section the relocation is contained in.
    uint64_t    Offset;     // Offset into the section for the relocation.
    uint32_t    Data;       // Second word of the raw macho relocation entry.
    int64_t     Addend;     // Addend encoded in the instruction itself, if any,
                            // plus the offset into the source section for
                            // the symbol once the relocation is resolvable.

    RelocationEntry(unsigned id, uint64_t offset, uint32_t data, int64_t addend)
      : SectionID(id), Offset(offset), Data(data), Addend(addend) {}
  };
  typedef SmallVector<RelocationEntry, 4> RelocationList;
  // Relocations to sections already loaded. Indexed by SectionID which is the
  // source of the address. The target where the address will be writen is
  // SectionID/Offset in the relocation itself.
  IndexedMap<RelocationList> Relocations;
  // Relocations to symbols that are not yet resolved. Must be external
  // relocations by definition. Indexed by symbol name.
  StringMap<RelocationList> UnresolvedRelocations;

  bool resolveRelocation(uint8_t *Address, uint64_t Value, bool isPCRel,
                         unsigned Type, unsigned Size, int64_t Addend);
  bool resolveX86_64Relocation(uintptr_t Address, uintptr_t Value, bool isPCRel,
                               unsigned Type, unsigned Size, int64_t Addend);
  bool resolveARMRelocation(uintptr_t Address, uintptr_t Value, bool isPCRel,
                            unsigned Type, unsigned Size, int64_t Addend);

  bool loadSegment32(const MachOObject *Obj,
                     const MachOObject::LoadCommandInfo *SegmentLCI,
                     const InMemoryStruct<macho::SymtabLoadCommand> &SymtabLC);
  bool loadSegment64(const MachOObject *Obj,
                     const MachOObject::LoadCommandInfo *SegmentLCI,
                     const InMemoryStruct<macho::SymtabLoadCommand> &SymtabLC);
  bool processSymbols32(const MachOObject *Obj,
                      SmallVectorImpl<unsigned> &SectionMap,
                      SmallVectorImpl<StringRef> &SymbolNames,
                      const InMemoryStruct<macho::SymtabLoadCommand> &SymtabLC);
  bool processSymbols64(const MachOObject *Obj,
                      SmallVectorImpl<unsigned> &SectionMap,
                      SmallVectorImpl<StringRef> &SymbolNames,
                      const InMemoryStruct<macho::SymtabLoadCommand> &SymtabLC);

  void resolveSymbol(StringRef Name);

public:
  RuntimeDyldMachO(RTDyldMemoryManager *mm) : RuntimeDyldImpl(mm) {}

  bool loadObject(MemoryBuffer *InputBuffer);

  void reassignSectionAddress(unsigned SectionID, uint64_t Addr);

  static bool isKnownFormat(const MemoryBuffer *InputBuffer);

  bool isCompatibleFormat(const MemoryBuffer *InputBuffer) const {
    return isKnownFormat(InputBuffer);
  }
};

} // end namespace llvm


#endif
