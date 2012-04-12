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
#include "llvm/Object/ObjectFile.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/system_error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/ADT/Triple.h"
#include <map>
#include "llvm/Support/Format.h"

using namespace llvm;
using namespace llvm::object;

namespace llvm {

class SectionEntry {
public:
  uint8_t* Address;
  size_t Size;
  uint64_t LoadAddress;   // For each section, the address it will be
                          // considered to live at for relocations. The same
                          // as the pointer to the above memory block for
                          // hosted JITs.
  uintptr_t StubOffset;   // It's used for architecturies with stub
                          // functions for far relocations like ARM.
  uintptr_t ObjAddress;   // Section address in object file. It's use for
                          // calculate MachO relocation addend
  SectionEntry(uint8_t* address, size_t size, uintptr_t stubOffset,
               uintptr_t objAddress)
    : Address(address), Size(size), LoadAddress((uintptr_t)address),
      StubOffset(stubOffset), ObjAddress(objAddress) {}
};

class RelocationEntry {
public:
  unsigned    SectionID;  // Section the relocation is contained in.
  uintptr_t   Offset;     // Offset into the section for the relocation.
  uint32_t    Data;       // Relocatino data. Including type of relocation
                          // and another flags and parameners from
  intptr_t    Addend;     // Addend encoded in the instruction itself, if any,
                          // plus the offset into the source section for
                          // the symbol once the relocation is resolvable.
  RelocationEntry(unsigned id, uint64_t offset, uint32_t data, int64_t addend)
    : SectionID(id), Offset(offset), Data(data), Addend(addend) {}
};

// Raw relocation data from object file
class ObjRelocationInfo {
public:
  unsigned  SectionID;
  uint64_t  Offset;
  SymbolRef Symbol;
  uint64_t  Type;
  int64_t   AdditionalInfo;
};

class RelocationValueRef {
public:
  unsigned  SectionID;
  intptr_t  Addend;
  const char *SymbolName;
  RelocationValueRef(): SectionID(0), Addend(0), SymbolName(0) {}

  inline bool operator==(const RelocationValueRef &Other) const {
    return std::memcmp(this, &Other, sizeof(RelocationValueRef)) == 0;
  }
  inline bool operator <(const RelocationValueRef &Other) const {
    return std::memcmp(this, &Other, sizeof(RelocationValueRef)) < 0;
  }
};

class RuntimeDyldImpl {
protected:
  // The MemoryManager to load objects into.
  RTDyldMemoryManager *MemMgr;

  // A list of emmitted sections.
  typedef SmallVector<SectionEntry, 64> SectionList;
  SectionList Sections;

  // Keep a map of sections from object file to the SectionID which
  // references it.
  typedef std::map<SectionRef, unsigned> ObjSectionToIDMap;

  // Master symbol table. As modules are loaded and external symbols are
  // resolved, their addresses are stored here as a SectionID/Offset pair.
  typedef std::pair<unsigned, uintptr_t> SymbolLoc;
  StringMap<SymbolLoc> SymbolTable;
  typedef DenseMap<const char*, SymbolLoc> LocalSymbolMap;

  // Keep a map of common symbols to their sizes
  typedef std::map<SymbolRef, unsigned> CommonSymbolMap;

  // For each symbol, keep a list of relocations based on it. Anytime
  // its address is reassigned (the JIT re-compiled the function, e.g.),
  // the relocations get re-resolved.
  // The symbol (or section) the relocation is sourced from is the Key
  // in the relocation list where it's stored.
  typedef SmallVector<RelocationEntry, 64> RelocationList;
  // Relocations to sections already loaded. Indexed by SectionID which is the
  // source of the address. The target where the address will be writen is
  // SectionID/Offset in the relocation itself.
  DenseMap<unsigned, RelocationList> Relocations;
  // Relocations to external symbols that are not yet resolved.
  // Indexed by symbol name.
  StringMap<RelocationList> SymbolRelocations;

  typedef std::map<RelocationValueRef, uintptr_t> StubMap;

  Triple::ArchType Arch;

  inline unsigned getMaxStubSize() {
    if (Arch == Triple::arm || Arch == Triple::thumb)
      return 8; // 32-bit instruction and 32-bit address
    else
      return 0;
  }

  bool HasError;
  std::string ErrorStr;

  // Set the error state and record an error string.
  bool Error(const Twine &Msg) {
    ErrorStr = Msg.str();
    HasError = true;
    return true;
  }

  uint8_t *getSectionAddress(unsigned SectionID) {
    return (uint8_t*)Sections[SectionID].Address;
  }

  /// \brief Emits a section containing common symbols.
  /// \return SectionID.
  unsigned emitCommonSymbols(const CommonSymbolMap &Map,
                             uint64_t TotalSize,
                             LocalSymbolMap &Symbols);

  /// \brief Emits section data from the object file to the MemoryManager.
  /// \param IsCode if it's true then allocateCodeSection() will be
  ///        used for emmits, else allocateDataSection() will be used.
  /// \return SectionID.
  unsigned emitSection(const SectionRef &Section, bool IsCode);

  /// \brief Find Section in LocalSections. If the secton is not found - emit
  ///        it and store in LocalSections.
  /// \param IsCode if it's true then allocateCodeSection() will be
  ///        used for emmits, else allocateDataSection() will be used.
  /// \return SectionID.
  unsigned findOrEmitSection(const SectionRef &Section, bool IsCode,
                             ObjSectionToIDMap &LocalSections);

  /// \brief If Value.SymbolName is NULL then store relocation to the
  ///        Relocations, else store it in the SymbolRelocations.
  void AddRelocation(const RelocationValueRef &Value, unsigned SectionID,
                     uintptr_t Offset, uint32_t RelType);

  /// \brief Emits long jump instruction to Addr.
  /// \return Pointer to the memory area for emitting target address.
  uint8_t* createStubFunction(uint8_t *Addr);

  /// \brief Resolves relocations from Relocs list with address from Value.
  void resolveRelocationList(const RelocationList &Relocs, uint64_t Value);
  void resolveRelocationEntry(const RelocationEntry &RE, uint64_t Value);

  /// \brief A object file specific relocation resolver
  /// \param Address Address to apply the relocation action
  /// \param Value Target symbol address to apply the relocation action
  /// \param Type object file specific relocation type
  /// \param Addend A constant addend used to compute the value to be stored
  ///        into the relocatable field
  virtual void resolveRelocation(uint8_t *LocalAddress,
                                 uint64_t FinalAddress,
                                 uint64_t Value,
                                 uint32_t Type,
                                 int64_t Addend) = 0;

  /// \brief Parses the object file relocation and store it to Relocations
  ///        or SymbolRelocations. Its depend from object file type.
  virtual void processRelocationRef(const ObjRelocationInfo &Rel,
                                    const ObjectFile &Obj,
                                    ObjSectionToIDMap &ObjSectionToID,
                                    LocalSymbolMap &Symbols, StubMap &Stubs) = 0;

  void resolveSymbols();
public:
  RuntimeDyldImpl(RTDyldMemoryManager *mm) : MemMgr(mm), HasError(false) {}

  virtual ~RuntimeDyldImpl();

  bool loadObject(const MemoryBuffer *InputBuffer);

  void *getSymbolAddress(StringRef Name) {
    // FIXME: Just look up as a function for now. Overly simple of course.
    // Work in progress.
    if (SymbolTable.find(Name) == SymbolTable.end())
      return 0;
    SymbolLoc Loc = SymbolTable.lookup(Name);
    return getSectionAddress(Loc.first) + Loc.second;
  }

  void resolveRelocations();

  void reassignSectionAddress(unsigned SectionID, uint64_t Addr);

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
