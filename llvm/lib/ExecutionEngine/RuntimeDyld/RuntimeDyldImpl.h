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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/ObjectImage.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/ExecutionEngine/RuntimeDyldChecker.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/SwapByteOrder.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <system_error>

using namespace llvm;
using namespace llvm::object;

namespace llvm {

class ObjectBuffer;
class Twine;

/// SectionEntry - represents a section emitted into memory by the dynamic
/// linker.
class SectionEntry {
public:
  /// Name - section name.
  StringRef Name;

  /// Address - address in the linker's memory where the section resides.
  uint8_t *Address;

  /// Size - section size. Doesn't include the stubs.
  size_t Size;

  /// LoadAddress - the address of the section in the target process's memory.
  /// Used for situations in which JIT-ed code is being executed in the address
  /// space of a separate process.  If the code executes in the same address
  /// space where it was JIT-ed, this just equals Address.
  uint64_t LoadAddress;

  /// StubOffset - used for architectures with stub functions for far
  /// relocations (like ARM).
  uintptr_t StubOffset;

  /// ObjAddress - address of the section in the in-memory object file.  Used
  /// for calculating relocations in some object formats (like MachO).
  uintptr_t ObjAddress;

  SectionEntry(StringRef name, uint8_t *address, size_t size,
               uintptr_t objAddress)
      : Name(name), Address(address), Size(size),
        LoadAddress((uintptr_t)address), StubOffset(size),
        ObjAddress(objAddress) {}
};

/// RelocationEntry - used to represent relocations internally in the dynamic
/// linker.
class RelocationEntry {
public:
  /// SectionID - the section this relocation points to.
  unsigned SectionID;

  /// Offset - offset into the section.
  uint64_t Offset;

  /// RelType - relocation type.
  uint32_t RelType;

  /// Addend - the relocation addend encoded in the instruction itself.  Also
  /// used to make a relocation section relative instead of symbol relative.
  int64_t Addend;

  struct SectionPair {
      uint32_t SectionA;
      uint32_t SectionB;
  };

  /// SymOffset - Section offset of the relocation entry's symbol (used for GOT
  /// lookup).
  union {
    uint64_t SymOffset;
    SectionPair Sections;
  };

  /// True if this is a PCRel relocation (MachO specific).
  bool IsPCRel;

  /// The size of this relocation (MachO specific).
  unsigned Size;

  RelocationEntry(unsigned id, uint64_t offset, uint32_t type, int64_t addend)
      : SectionID(id), Offset(offset), RelType(type), Addend(addend),
        SymOffset(0), IsPCRel(false), Size(0) {}

  RelocationEntry(unsigned id, uint64_t offset, uint32_t type, int64_t addend,
                  uint64_t symoffset)
      : SectionID(id), Offset(offset), RelType(type), Addend(addend),
        SymOffset(symoffset), IsPCRel(false), Size(0) {}

  RelocationEntry(unsigned id, uint64_t offset, uint32_t type, int64_t addend,
                  bool IsPCRel, unsigned Size)
      : SectionID(id), Offset(offset), RelType(type), Addend(addend),
        SymOffset(0), IsPCRel(IsPCRel), Size(Size) {}

  RelocationEntry(unsigned id, uint64_t offset, uint32_t type, int64_t addend,
                  unsigned SectionA, uint64_t SectionAOffset, unsigned SectionB,
                  uint64_t SectionBOffset, bool IsPCRel, unsigned Size)
      : SectionID(id), Offset(offset), RelType(type),
        Addend(SectionAOffset - SectionBOffset + addend), IsPCRel(IsPCRel),
        Size(Size) {
    Sections.SectionA = SectionA;
    Sections.SectionB = SectionB;
  }
};

class RelocationValueRef {
public:
  unsigned SectionID;
  uint64_t Offset;
  int64_t Addend;
  const char *SymbolName;
  RelocationValueRef() : SectionID(0), Offset(0), Addend(0),
                         SymbolName(nullptr) {}

  inline bool operator==(const RelocationValueRef &Other) const {
    return SectionID == Other.SectionID && Offset == Other.Offset &&
           Addend == Other.Addend && SymbolName == Other.SymbolName;
  }
  inline bool operator<(const RelocationValueRef &Other) const {
    if (SectionID != Other.SectionID)
      return SectionID < Other.SectionID;
    if (Offset != Other.Offset)
      return Offset < Other.Offset;
    if (Addend != Other.Addend)
      return Addend < Other.Addend;
    return SymbolName < Other.SymbolName;
  }
};

class RuntimeDyldImpl {
  friend class RuntimeDyldCheckerImpl;
private:

  uint64_t getAnySymbolRemoteAddress(StringRef Symbol) {
    if (uint64_t InternalSymbolAddr = getSymbolLoadAddress(Symbol))
      return InternalSymbolAddr;
    return MemMgr->getSymbolAddress(Symbol);
  }

protected:
  // The MemoryManager to load objects into.
  RTDyldMemoryManager *MemMgr;

  // Attached RuntimeDyldChecker instance. Null if no instance attached.
  RuntimeDyldCheckerImpl *Checker;

  // A list of all sections emitted by the dynamic linker.  These sections are
  // referenced in the code by means of their index in this list - SectionID.
  typedef SmallVector<SectionEntry, 64> SectionList;
  SectionList Sections;

  typedef unsigned SID; // Type for SectionIDs
#define RTDYLD_INVALID_SECTION_ID ((SID)(-1))

  // Keep a map of sections from object file to the SectionID which
  // references it.
  typedef std::map<SectionRef, unsigned> ObjSectionToIDMap;

  // A global symbol table for symbols from all loaded modules.  Maps the
  // symbol name to a (SectionID, offset in section) pair.
  typedef std::pair<unsigned, uintptr_t> SymbolLoc;
  typedef StringMap<SymbolLoc> SymbolTableMap;
  SymbolTableMap GlobalSymbolTable;

  // Pair representing the size and alignment requirement for a common symbol.
  typedef std::pair<unsigned, unsigned> CommonSymbolInfo;
  // Keep a map of common symbols to their info pairs
  typedef std::map<SymbolRef, CommonSymbolInfo> CommonSymbolMap;

  // For each symbol, keep a list of relocations based on it. Anytime
  // its address is reassigned (the JIT re-compiled the function, e.g.),
  // the relocations get re-resolved.
  // The symbol (or section) the relocation is sourced from is the Key
  // in the relocation list where it's stored.
  typedef SmallVector<RelocationEntry, 64> RelocationList;
  // Relocations to sections already loaded. Indexed by SectionID which is the
  // source of the address. The target where the address will be written is
  // SectionID/Offset in the relocation itself.
  DenseMap<unsigned, RelocationList> Relocations;

  // Relocations to external symbols that are not yet resolved.  Symbols are
  // external when they aren't found in the global symbol table of all loaded
  // modules.  This map is indexed by symbol name.
  StringMap<RelocationList> ExternalSymbolRelocations;


  typedef std::map<RelocationValueRef, uintptr_t> StubMap;

  Triple::ArchType Arch;
  bool IsTargetLittleEndian;

  // True if all sections should be passed to the memory manager, false if only
  // sections containing relocations should be. Defaults to 'false'.
  bool ProcessAllSections;

  // This mutex prevents simultaneously loading objects from two different
  // threads.  This keeps us from having to protect individual data structures
  // and guarantees that section allocation requests to the memory manager
  // won't be interleaved between modules.  It is also used in mapSectionAddress
  // and resolveRelocations to protect write access to internal data structures.
  //
  // loadObject may be called on the same thread during the handling of of
  // processRelocations, and that's OK.  The handling of the relocation lists
  // is written in such a way as to work correctly if new elements are added to
  // the end of the list while the list is being processed.
  sys::Mutex lock;

  virtual unsigned getMaxStubSize() = 0;
  virtual unsigned getStubAlignment() = 0;

  bool HasError;
  std::string ErrorStr;

  // Set the error state and record an error string.
  bool Error(const Twine &Msg) {
    ErrorStr = Msg.str();
    HasError = true;
    return true;
  }

  uint64_t getSectionLoadAddress(unsigned SectionID) {
    return Sections[SectionID].LoadAddress;
  }

  uint8_t *getSectionAddress(unsigned SectionID) {
    return (uint8_t *)Sections[SectionID].Address;
  }

  void writeInt16BE(uint8_t *Addr, uint16_t Value) {
    if (IsTargetLittleEndian)
      sys::swapByteOrder(Value);
    *Addr       = (Value >> 8) & 0xFF;
    *(Addr + 1) = Value & 0xFF;
  }

  void writeInt32BE(uint8_t *Addr, uint32_t Value) {
    if (IsTargetLittleEndian)
      sys::swapByteOrder(Value);
    *Addr       = (Value >> 24) & 0xFF;
    *(Addr + 1) = (Value >> 16) & 0xFF;
    *(Addr + 2) = (Value >> 8) & 0xFF;
    *(Addr + 3) = Value & 0xFF;
  }

  void writeInt64BE(uint8_t *Addr, uint64_t Value) {
    if (IsTargetLittleEndian)
      sys::swapByteOrder(Value);
    *Addr       = (Value >> 56) & 0xFF;
    *(Addr + 1) = (Value >> 48) & 0xFF;
    *(Addr + 2) = (Value >> 40) & 0xFF;
    *(Addr + 3) = (Value >> 32) & 0xFF;
    *(Addr + 4) = (Value >> 24) & 0xFF;
    *(Addr + 5) = (Value >> 16) & 0xFF;
    *(Addr + 6) = (Value >> 8) & 0xFF;
    *(Addr + 7) = Value & 0xFF;
  }

  /// \brief Given the common symbols discovered in the object file, emit a
  /// new section for them and update the symbol mappings in the object and
  /// symbol table.
  void emitCommonSymbols(ObjectImage &Obj, const CommonSymbolMap &CommonSymbols,
                         uint64_t TotalSize, SymbolTableMap &SymbolTable);

  /// \brief Emits section data from the object file to the MemoryManager.
  /// \param IsCode if it's true then allocateCodeSection() will be
  ///        used for emits, else allocateDataSection() will be used.
  /// \return SectionID.
  unsigned emitSection(ObjectImage &Obj, const SectionRef &Section,
                       bool IsCode);

  /// \brief Find Section in LocalSections. If the secton is not found - emit
  ///        it and store in LocalSections.
  /// \param IsCode if it's true then allocateCodeSection() will be
  ///        used for emmits, else allocateDataSection() will be used.
  /// \return SectionID.
  unsigned findOrEmitSection(ObjectImage &Obj, const SectionRef &Section,
                             bool IsCode, ObjSectionToIDMap &LocalSections);

  // \brief Add a relocation entry that uses the given section.
  void addRelocationForSection(const RelocationEntry &RE, unsigned SectionID);

  // \brief Add a relocation entry that uses the given symbol.  This symbol may
  // be found in the global symbol table, or it may be external.
  void addRelocationForSymbol(const RelocationEntry &RE, StringRef SymbolName);

  /// \brief Emits long jump instruction to Addr.
  /// \return Pointer to the memory area for emitting target address.
  uint8_t *createStubFunction(uint8_t *Addr, unsigned AbiVariant = 0);

  /// \brief Resolves relocations from Relocs list with address from Value.
  void resolveRelocationList(const RelocationList &Relocs, uint64_t Value);

  /// \brief A object file specific relocation resolver
  /// \param RE The relocation to be resolved
  /// \param Value Target symbol address to apply the relocation action
  virtual void resolveRelocation(const RelocationEntry &RE, uint64_t Value) = 0;

  /// \brief Parses one or more object file relocations (some object files use
  ///        relocation pairs) and stores it to Relocations or SymbolRelocations
  ///        (this depends on the object file type).
  /// \return Iterator to the next relocation that needs to be parsed.
  virtual relocation_iterator
  processRelocationRef(unsigned SectionID, relocation_iterator RelI,
                       ObjectImage &Obj, ObjSectionToIDMap &ObjSectionToID,
                       const SymbolTableMap &Symbols, StubMap &Stubs) = 0;

  /// \brief Resolve relocations to external symbols.
  void resolveExternalSymbols();

  /// \brief Update GOT entries for external symbols.
  // The base class does nothing.  ELF overrides this.
  virtual void updateGOTEntries(StringRef Name, uint64_t Addr) {}

  // \brief Compute an upper bound of the memory that is required to load all
  // sections
  void computeTotalAllocSize(ObjectImage &Obj, uint64_t &CodeSize,
                             uint64_t &DataSizeRO, uint64_t &DataSizeRW);

  // \brief Compute the stub buffer size required for a section
  unsigned computeSectionStubBufSize(ObjectImage &Obj,
                                     const SectionRef &Section);

public:
  RuntimeDyldImpl(RTDyldMemoryManager *mm)
    : MemMgr(mm), Checker(nullptr), ProcessAllSections(false), HasError(false) {
  }

  virtual ~RuntimeDyldImpl();

  void setProcessAllSections(bool ProcessAllSections) {
    this->ProcessAllSections = ProcessAllSections;
  }

  void setRuntimeDyldChecker(RuntimeDyldCheckerImpl *Checker) {
    this->Checker = Checker;
  }

  ObjectImage *loadObject(ObjectImage *InputObject);

  uint8_t* getSymbolAddress(StringRef Name) {
    // FIXME: Just look up as a function for now. Overly simple of course.
    // Work in progress.
    SymbolTableMap::const_iterator pos = GlobalSymbolTable.find(Name);
    if (pos == GlobalSymbolTable.end())
      return nullptr;
    SymbolLoc Loc = pos->second;
    return getSectionAddress(Loc.first) + Loc.second;
  }

  uint64_t getSymbolLoadAddress(StringRef Name) {
    // FIXME: Just look up as a function for now. Overly simple of course.
    // Work in progress.
    SymbolTableMap::const_iterator pos = GlobalSymbolTable.find(Name);
    if (pos == GlobalSymbolTable.end())
      return 0;
    SymbolLoc Loc = pos->second;
    return getSectionLoadAddress(Loc.first) + Loc.second;
  }

  void resolveRelocations();

  void reassignSectionAddress(unsigned SectionID, uint64_t Addr);

  void mapSectionAddress(const void *LocalAddress, uint64_t TargetAddress);

  // Is the linker in an error state?
  bool hasError() { return HasError; }

  // Mark the error condition as handled and continue.
  void clearError() { HasError = false; }

  // Get the error message.
  StringRef getErrorString() { return ErrorStr; }

  virtual bool isCompatibleFormat(const ObjectBuffer *Buffer) const = 0;
  virtual bool isCompatibleFile(const ObjectFile *Obj) const = 0;

  virtual void registerEHFrames();

  virtual void deregisterEHFrames();

  virtual void finalizeLoad(ObjectImage &ObjImg, ObjSectionToIDMap &SectionMap) {}
};

} // end namespace llvm

#endif
