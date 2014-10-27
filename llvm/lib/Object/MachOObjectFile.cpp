//===- MachOObjectFile.cpp - Mach-O object file binding ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MachOObjectFile class, which binds the MachOObject
// class to the generic ObjectFile wrapper.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/MachO.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MachO.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
#include <cstring>
#include <limits>

using namespace llvm;
using namespace object;

namespace {
  struct section_base {
    char sectname[16];
    char segname[16];
  };
}

template<typename T>
static T getStruct(const MachOObjectFile *O, const char *P) {
  T Cmd;
  memcpy(&Cmd, P, sizeof(T));
  if (O->isLittleEndian() != sys::IsLittleEndianHost)
    MachO::swapStruct(Cmd);
  return Cmd;
}

static uint32_t
getSegmentLoadCommandNumSections(const MachOObjectFile *O,
                                 const MachOObjectFile::LoadCommandInfo &L) {
  if (O->is64Bit()) {
    MachO::segment_command_64 S = O->getSegment64LoadCommand(L);
    return S.nsects;
  }
  MachO::segment_command S = O->getSegmentLoadCommand(L);
  return S.nsects;
}

static bool isPageZeroSegment(const MachOObjectFile *O,
                              const MachOObjectFile::LoadCommandInfo &L) {
  if (O->is64Bit()) {
    MachO::segment_command_64 S = O->getSegment64LoadCommand(L);
    return StringRef("__PAGEZERO").equals(S.segname);
  }
  MachO::segment_command S = O->getSegmentLoadCommand(L);
  return StringRef("__PAGEZERO").equals(S.segname);
}


static const char *
getSectionPtr(const MachOObjectFile *O, MachOObjectFile::LoadCommandInfo L,
              unsigned Sec) {
  uintptr_t CommandAddr = reinterpret_cast<uintptr_t>(L.Ptr);

  bool Is64 = O->is64Bit();
  unsigned SegmentLoadSize = Is64 ? sizeof(MachO::segment_command_64) :
                                    sizeof(MachO::segment_command);
  unsigned SectionSize = Is64 ? sizeof(MachO::section_64) :
                                sizeof(MachO::section);

  uintptr_t SectionAddr = CommandAddr + SegmentLoadSize + Sec * SectionSize;
  return reinterpret_cast<const char*>(SectionAddr);
}

static const char *getPtr(const MachOObjectFile *O, size_t Offset) {
  return O->getData().substr(Offset, 1).data();
}

static MachO::nlist_base
getSymbolTableEntryBase(const MachOObjectFile *O, DataRefImpl DRI) {
  const char *P = reinterpret_cast<const char *>(DRI.p);
  return getStruct<MachO::nlist_base>(O, P);
}

static StringRef parseSegmentOrSectionName(const char *P) {
  if (P[15] == 0)
    // Null terminated.
    return P;
  // Not null terminated, so this is a 16 char string.
  return StringRef(P, 16);
}

// Helper to advance a section or symbol iterator multiple increments at a time.
template<class T>
static void advance(T &it, size_t Val) {
  while (Val--)
    ++it;
}

static unsigned getCPUType(const MachOObjectFile *O) {
  return O->getHeader().cputype;
}

static void printRelocationTargetName(const MachOObjectFile *O,
                                      const MachO::any_relocation_info &RE,
                                      raw_string_ostream &fmt) {
  bool IsScattered = O->isRelocationScattered(RE);

  // Target of a scattered relocation is an address.  In the interest of
  // generating pretty output, scan through the symbol table looking for a
  // symbol that aligns with that address.  If we find one, print it.
  // Otherwise, we just print the hex address of the target.
  if (IsScattered) {
    uint32_t Val = O->getPlainRelocationSymbolNum(RE);

    for (const SymbolRef &Symbol : O->symbols()) {
      std::error_code ec;
      uint64_t Addr;
      StringRef Name;

      if ((ec = Symbol.getAddress(Addr)))
        report_fatal_error(ec.message());
      if (Addr != Val)
        continue;
      if ((ec = Symbol.getName(Name)))
        report_fatal_error(ec.message());
      fmt << Name;
      return;
    }

    // If we couldn't find a symbol that this relocation refers to, try
    // to find a section beginning instead.
    for (const SectionRef &Section : O->sections()) {
      std::error_code ec;

      StringRef Name;
      uint64_t Addr = Section.getAddress();
      if (Addr != Val)
        continue;
      if ((ec = Section.getName(Name)))
        report_fatal_error(ec.message());
      fmt << Name;
      return;
    }

    fmt << format("0x%x", Val);
    return;
  }

  StringRef S;
  bool isExtern = O->getPlainRelocationExternal(RE);
  uint64_t Val = O->getPlainRelocationSymbolNum(RE);

  if (isExtern) {
    symbol_iterator SI = O->symbol_begin();
    advance(SI, Val);
    SI->getName(S);
  } else {
    section_iterator SI = O->section_begin();
    // Adjust for the fact that sections are 1-indexed.
    advance(SI, Val - 1);
    SI->getName(S);
  }

  fmt << S;
}

static uint32_t
getPlainRelocationAddress(const MachO::any_relocation_info &RE) {
  return RE.r_word0;
}

static unsigned
getScatteredRelocationAddress(const MachO::any_relocation_info &RE) {
  return RE.r_word0 & 0xffffff;
}

static bool getPlainRelocationPCRel(const MachOObjectFile *O,
                                    const MachO::any_relocation_info &RE) {
  if (O->isLittleEndian())
    return (RE.r_word1 >> 24) & 1;
  return (RE.r_word1 >> 7) & 1;
}

static bool
getScatteredRelocationPCRel(const MachOObjectFile *O,
                            const MachO::any_relocation_info &RE) {
  return (RE.r_word0 >> 30) & 1;
}

static unsigned getPlainRelocationLength(const MachOObjectFile *O,
                                         const MachO::any_relocation_info &RE) {
  if (O->isLittleEndian())
    return (RE.r_word1 >> 25) & 3;
  return (RE.r_word1 >> 5) & 3;
}

static unsigned
getScatteredRelocationLength(const MachO::any_relocation_info &RE) {
  return (RE.r_word0 >> 28) & 3;
}

static unsigned getPlainRelocationType(const MachOObjectFile *O,
                                       const MachO::any_relocation_info &RE) {
  if (O->isLittleEndian())
    return RE.r_word1 >> 28;
  return RE.r_word1 & 0xf;
}

static unsigned
getScatteredRelocationType(const MachO::any_relocation_info &RE) {
  return (RE.r_word0 >> 24) & 0xf;
}

static uint32_t getSectionFlags(const MachOObjectFile *O,
                                DataRefImpl Sec) {
  if (O->is64Bit()) {
    MachO::section_64 Sect = O->getSection64(Sec);
    return Sect.flags;
  }
  MachO::section Sect = O->getSection(Sec);
  return Sect.flags;
}

MachOObjectFile::MachOObjectFile(MemoryBufferRef Object, bool IsLittleEndian,
                                 bool Is64bits, std::error_code &EC)
    : ObjectFile(getMachOType(IsLittleEndian, Is64bits), Object),
      SymtabLoadCmd(nullptr), DysymtabLoadCmd(nullptr),
      DataInCodeLoadCmd(nullptr), DyldInfoLoadCmd(nullptr),
      UuidLoadCmd(nullptr), HasPageZeroSegment(false) {
  uint32_t LoadCommandCount = this->getHeader().ncmds;
  MachO::LoadCommandType SegmentLoadType = is64Bit() ?
    MachO::LC_SEGMENT_64 : MachO::LC_SEGMENT;

  MachOObjectFile::LoadCommandInfo Load = getFirstLoadCommandInfo();
  for (unsigned I = 0; ; ++I) {
    if (Load.C.cmd == MachO::LC_SYMTAB) {
      assert(!SymtabLoadCmd && "Multiple symbol tables");
      SymtabLoadCmd = Load.Ptr;
    } else if (Load.C.cmd == MachO::LC_DYSYMTAB) {
      assert(!DysymtabLoadCmd && "Multiple dynamic symbol tables");
      DysymtabLoadCmd = Load.Ptr;
    } else if (Load.C.cmd == MachO::LC_DATA_IN_CODE) {
      assert(!DataInCodeLoadCmd && "Multiple data in code tables");
      DataInCodeLoadCmd = Load.Ptr;
    } else if (Load.C.cmd == MachO::LC_DYLD_INFO || 
               Load.C.cmd == MachO::LC_DYLD_INFO_ONLY) {
      assert(!DyldInfoLoadCmd && "Multiple dyldinfo load commands");
      DyldInfoLoadCmd = Load.Ptr;
    } else if (Load.C.cmd == MachO::LC_UUID) {
      assert(!UuidLoadCmd && "Multiple UUID load commands");
      UuidLoadCmd = Load.Ptr;
    } else if (Load.C.cmd == SegmentLoadType) {
      uint32_t NumSections = getSegmentLoadCommandNumSections(this, Load);
      for (unsigned J = 0; J < NumSections; ++J) {
        const char *Sec = getSectionPtr(this, Load, J);
        Sections.push_back(Sec);
      }
      if (isPageZeroSegment(this, Load))
        HasPageZeroSegment = true;
    } else if (Load.C.cmd == MachO::LC_LOAD_DYLIB ||
               Load.C.cmd == MachO::LC_LOAD_WEAK_DYLIB ||
               Load.C.cmd == MachO::LC_LAZY_LOAD_DYLIB ||
               Load.C.cmd == MachO::LC_REEXPORT_DYLIB ||
               Load.C.cmd == MachO::LC_LOAD_UPWARD_DYLIB) {
      Libraries.push_back(Load.Ptr);
    }

    if (I == LoadCommandCount - 1)
      break;
    else
      Load = getNextLoadCommandInfo(Load);
  }
}

void MachOObjectFile::moveSymbolNext(DataRefImpl &Symb) const {
  unsigned SymbolTableEntrySize = is64Bit() ?
    sizeof(MachO::nlist_64) :
    sizeof(MachO::nlist);
  Symb.p += SymbolTableEntrySize;
}

std::error_code MachOObjectFile::getSymbolName(DataRefImpl Symb,
                                               StringRef &Res) const {
  StringRef StringTable = getStringTableData();
  MachO::nlist_base Entry = getSymbolTableEntryBase(this, Symb);
  const char *Start = &StringTable.data()[Entry.n_strx];
  Res = StringRef(Start);
  return object_error::success;
}

// getIndirectName() returns the name of the alias'ed symbol who's string table
// index is in the n_value field.
std::error_code MachOObjectFile::getIndirectName(DataRefImpl Symb,
                                                 StringRef &Res) const {
  StringRef StringTable = getStringTableData();
  uint64_t NValue;
  if (is64Bit()) {
    MachO::nlist_64 Entry = getSymbol64TableEntry(Symb);
    NValue = Entry.n_value;
    if ((Entry.n_type & MachO::N_TYPE) != MachO::N_INDR)
      return object_error::parse_failed;
  } else {
    MachO::nlist Entry = getSymbolTableEntry(Symb);
    NValue = Entry.n_value;
    if ((Entry.n_type & MachO::N_TYPE) != MachO::N_INDR)
      return object_error::parse_failed;
  }
  if (NValue >= StringTable.size())
    return object_error::parse_failed;
  const char *Start = &StringTable.data()[NValue];
  Res = StringRef(Start);
  return object_error::success;
}

std::error_code MachOObjectFile::getSymbolAddress(DataRefImpl Symb,
                                                  uint64_t &Res) const {
  if (is64Bit()) {
    MachO::nlist_64 Entry = getSymbol64TableEntry(Symb);
    if ((Entry.n_type & MachO::N_TYPE) == MachO::N_UNDF &&
        Entry.n_value == 0)
      Res = UnknownAddressOrSize;
    else
      Res = Entry.n_value;
  } else {
    MachO::nlist Entry = getSymbolTableEntry(Symb);
    if ((Entry.n_type & MachO::N_TYPE) == MachO::N_UNDF &&
        Entry.n_value == 0)
      Res = UnknownAddressOrSize;
    else
      Res = Entry.n_value;
  }
  return object_error::success;
}

std::error_code MachOObjectFile::getSymbolAlignment(DataRefImpl DRI,
                                                    uint32_t &Result) const {
  uint32_t flags = getSymbolFlags(DRI);
  if (flags & SymbolRef::SF_Common) {
    MachO::nlist_base Entry = getSymbolTableEntryBase(this, DRI);
    Result = 1 << MachO::GET_COMM_ALIGN(Entry.n_desc);
  } else {
    Result = 0;
  }
  return object_error::success;
}

std::error_code MachOObjectFile::getSymbolSize(DataRefImpl DRI,
                                               uint64_t &Result) const {
  uint64_t BeginOffset;
  uint64_t EndOffset = 0;
  uint8_t SectionIndex;

  MachO::nlist_base Entry = getSymbolTableEntryBase(this, DRI);
  uint64_t Value;
  getSymbolAddress(DRI, Value);
  if (Value == UnknownAddressOrSize) {
    Result = UnknownAddressOrSize;
    return object_error::success;
  }

  BeginOffset = Value;

  SectionIndex = Entry.n_sect;
  if (!SectionIndex) {
    uint32_t flags = getSymbolFlags(DRI);
    if (flags & SymbolRef::SF_Common)
      Result = Value;
    else
      Result = UnknownAddressOrSize;
    return object_error::success;
  }
  // Unfortunately symbols are unsorted so we need to touch all
  // symbols from load command
  for (const SymbolRef &Symbol : symbols()) {
    DataRefImpl DRI = Symbol.getRawDataRefImpl();
    Entry = getSymbolTableEntryBase(this, DRI);
    getSymbolAddress(DRI, Value);
    if (Value == UnknownAddressOrSize)
      continue;
    if (Entry.n_sect == SectionIndex && Value > BeginOffset)
      if (!EndOffset || Value < EndOffset)
        EndOffset = Value;
  }
  if (!EndOffset) {
    DataRefImpl Sec;
    Sec.d.a = SectionIndex-1;
    uint64_t Size = getSectionSize(Sec);
    EndOffset = getSectionAddress(Sec);
    EndOffset += Size;
  }
  Result = EndOffset - BeginOffset;
  return object_error::success;
}

std::error_code MachOObjectFile::getSymbolType(DataRefImpl Symb,
                                               SymbolRef::Type &Res) const {
  MachO::nlist_base Entry = getSymbolTableEntryBase(this, Symb);
  uint8_t n_type = Entry.n_type;

  Res = SymbolRef::ST_Other;

  // If this is a STAB debugging symbol, we can do nothing more.
  if (n_type & MachO::N_STAB) {
    Res = SymbolRef::ST_Debug;
    return object_error::success;
  }

  switch (n_type & MachO::N_TYPE) {
    case MachO::N_UNDF :
      Res = SymbolRef::ST_Unknown;
      break;
    case MachO::N_SECT :
      Res = SymbolRef::ST_Function;
      break;
  }
  return object_error::success;
}

uint32_t MachOObjectFile::getSymbolFlags(DataRefImpl DRI) const {
  MachO::nlist_base Entry = getSymbolTableEntryBase(this, DRI);

  uint8_t MachOType = Entry.n_type;
  uint16_t MachOFlags = Entry.n_desc;

  uint32_t Result = SymbolRef::SF_None;

  if ((MachOType & MachO::N_TYPE) == MachO::N_UNDF)
    Result |= SymbolRef::SF_Undefined;

  if ((MachOType & MachO::N_TYPE) == MachO::N_INDR)
    Result |= SymbolRef::SF_Indirect;

  if (MachOType & MachO::N_STAB)
    Result |= SymbolRef::SF_FormatSpecific;

  if (MachOType & MachO::N_EXT) {
    Result |= SymbolRef::SF_Global;
    if ((MachOType & MachO::N_TYPE) == MachO::N_UNDF) {
      uint64_t Value;
      getSymbolAddress(DRI, Value);
      if (Value && Value != UnknownAddressOrSize)
        Result |= SymbolRef::SF_Common;
    }
  }

  if (MachOFlags & (MachO::N_WEAK_REF | MachO::N_WEAK_DEF))
    Result |= SymbolRef::SF_Weak;

  if (MachOFlags & (MachO::N_ARM_THUMB_DEF))
    Result |= SymbolRef::SF_Thumb;

  if ((MachOType & MachO::N_TYPE) == MachO::N_ABS)
    Result |= SymbolRef::SF_Absolute;

  return Result;
}

std::error_code MachOObjectFile::getSymbolSection(DataRefImpl Symb,
                                                  section_iterator &Res) const {
  MachO::nlist_base Entry = getSymbolTableEntryBase(this, Symb);
  uint8_t index = Entry.n_sect;

  if (index == 0) {
    Res = section_end();
  } else {
    DataRefImpl DRI;
    DRI.d.a = index - 1;
    Res = section_iterator(SectionRef(DRI, this));
  }

  return object_error::success;
}

void MachOObjectFile::moveSectionNext(DataRefImpl &Sec) const {
  Sec.d.a++;
}

std::error_code MachOObjectFile::getSectionName(DataRefImpl Sec,
                                                StringRef &Result) const {
  ArrayRef<char> Raw = getSectionRawName(Sec);
  Result = parseSegmentOrSectionName(Raw.data());
  return object_error::success;
}

uint64_t MachOObjectFile::getSectionAddress(DataRefImpl Sec) const {
  if (is64Bit())
    return getSection64(Sec).addr;
  return getSection(Sec).addr;
}

uint64_t MachOObjectFile::getSectionSize(DataRefImpl Sec) const {
  if (is64Bit())
    return getSection64(Sec).size;
  return getSection(Sec).size;
}

std::error_code MachOObjectFile::getSectionContents(DataRefImpl Sec,
                                                    StringRef &Res) const {
  uint32_t Offset;
  uint64_t Size;

  if (is64Bit()) {
    MachO::section_64 Sect = getSection64(Sec);
    Offset = Sect.offset;
    Size = Sect.size;
  } else {
    MachO::section Sect = getSection(Sec);
    Offset = Sect.offset;
    Size = Sect.size;
  }

  Res = this->getData().substr(Offset, Size);
  return object_error::success;
}

uint64_t MachOObjectFile::getSectionAlignment(DataRefImpl Sec) const {
  uint32_t Align;
  if (is64Bit()) {
    MachO::section_64 Sect = getSection64(Sec);
    Align = Sect.align;
  } else {
    MachO::section Sect = getSection(Sec);
    Align = Sect.align;
  }

  return uint64_t(1) << Align;
}

bool MachOObjectFile::isSectionText(DataRefImpl Sec) const {
  uint32_t Flags = getSectionFlags(this, Sec);
  return Flags & MachO::S_ATTR_PURE_INSTRUCTIONS;
}

bool MachOObjectFile::isSectionData(DataRefImpl Sec) const {
  uint32_t Flags = getSectionFlags(this, Sec);
  unsigned SectionType = Flags & MachO::SECTION_TYPE;
  return !(Flags & MachO::S_ATTR_PURE_INSTRUCTIONS) &&
         !(SectionType == MachO::S_ZEROFILL ||
           SectionType == MachO::S_GB_ZEROFILL);
}

bool MachOObjectFile::isSectionBSS(DataRefImpl Sec) const {
  uint32_t Flags = getSectionFlags(this, Sec);
  unsigned SectionType = Flags & MachO::SECTION_TYPE;
  return !(Flags & MachO::S_ATTR_PURE_INSTRUCTIONS) &&
         (SectionType == MachO::S_ZEROFILL ||
          SectionType == MachO::S_GB_ZEROFILL);
}

bool MachOObjectFile::isSectionRequiredForExecution(DataRefImpl Sect) const {
  // FIXME: Unimplemented.
  return true;
}

bool MachOObjectFile::isSectionVirtual(DataRefImpl Sec) const {
  // FIXME: Unimplemented.
  return false;
}

bool MachOObjectFile::isSectionZeroInit(DataRefImpl Sec) const {
  uint32_t Flags = getSectionFlags(this, Sec);
  unsigned SectionType = Flags & MachO::SECTION_TYPE;
  return SectionType == MachO::S_ZEROFILL ||
         SectionType == MachO::S_GB_ZEROFILL;
}

bool MachOObjectFile::isSectionReadOnlyData(DataRefImpl Sec) const {
  // Consider using the code from isSectionText to look for __const sections.
  // Alternately, emit S_ATTR_PURE_INSTRUCTIONS and/or S_ATTR_SOME_INSTRUCTIONS
  // to use section attributes to distinguish code from data.

  // FIXME: Unimplemented.
  return false;
}

bool MachOObjectFile::sectionContainsSymbol(DataRefImpl Sec,
                                            DataRefImpl Symb) const {
  SymbolRef::Type ST;
  this->getSymbolType(Symb, ST);
  if (ST == SymbolRef::ST_Unknown)
    return false;

  uint64_t SectBegin = getSectionAddress(Sec);
  uint64_t SectEnd = getSectionSize(Sec);
  SectEnd += SectBegin;

  uint64_t SymAddr;
  getSymbolAddress(Symb, SymAddr);
  return (SymAddr >= SectBegin) && (SymAddr < SectEnd);
}

relocation_iterator MachOObjectFile::section_rel_begin(DataRefImpl Sec) const {
  DataRefImpl Ret;
  Ret.d.a = Sec.d.a;
  Ret.d.b = 0;
  return relocation_iterator(RelocationRef(Ret, this));
}

relocation_iterator
MachOObjectFile::section_rel_end(DataRefImpl Sec) const {
  uint32_t Num;
  if (is64Bit()) {
    MachO::section_64 Sect = getSection64(Sec);
    Num = Sect.nreloc;
  } else {
    MachO::section Sect = getSection(Sec);
    Num = Sect.nreloc;
  }

  DataRefImpl Ret;
  Ret.d.a = Sec.d.a;
  Ret.d.b = Num;
  return relocation_iterator(RelocationRef(Ret, this));
}

void MachOObjectFile::moveRelocationNext(DataRefImpl &Rel) const {
  ++Rel.d.b;
}

std::error_code MachOObjectFile::getRelocationAddress(DataRefImpl Rel,
                                                      uint64_t &Res) const {
  uint64_t Offset;
  getRelocationOffset(Rel, Offset);

  DataRefImpl Sec;
  Sec.d.a = Rel.d.a;
  uint64_t SecAddress = getSectionAddress(Sec);
  Res = SecAddress + Offset;
  return object_error::success;
}

std::error_code MachOObjectFile::getRelocationOffset(DataRefImpl Rel,
                                                     uint64_t &Res) const {
  assert(getHeader().filetype == MachO::MH_OBJECT &&
         "Only implemented for MH_OBJECT");
  MachO::any_relocation_info RE = getRelocation(Rel);
  Res = getAnyRelocationAddress(RE);
  return object_error::success;
}

symbol_iterator
MachOObjectFile::getRelocationSymbol(DataRefImpl Rel) const {
  MachO::any_relocation_info RE = getRelocation(Rel);
  if (isRelocationScattered(RE))
    return symbol_end();

  uint32_t SymbolIdx = getPlainRelocationSymbolNum(RE);
  bool isExtern = getPlainRelocationExternal(RE);
  if (!isExtern)
    return symbol_end();

  MachO::symtab_command S = getSymtabLoadCommand();
  unsigned SymbolTableEntrySize = is64Bit() ?
    sizeof(MachO::nlist_64) :
    sizeof(MachO::nlist);
  uint64_t Offset = S.symoff + SymbolIdx * SymbolTableEntrySize;
  DataRefImpl Sym;
  Sym.p = reinterpret_cast<uintptr_t>(getPtr(this, Offset));
  return symbol_iterator(SymbolRef(Sym, this));
}

std::error_code MachOObjectFile::getRelocationType(DataRefImpl Rel,
                                                   uint64_t &Res) const {
  MachO::any_relocation_info RE = getRelocation(Rel);
  Res = getAnyRelocationType(RE);
  return object_error::success;
}

std::error_code
MachOObjectFile::getRelocationTypeName(DataRefImpl Rel,
                                       SmallVectorImpl<char> &Result) const {
  StringRef res;
  uint64_t RType;
  getRelocationType(Rel, RType);

  unsigned Arch = this->getArch();

  switch (Arch) {
    case Triple::x86: {
      static const char *const Table[] =  {
        "GENERIC_RELOC_VANILLA",
        "GENERIC_RELOC_PAIR",
        "GENERIC_RELOC_SECTDIFF",
        "GENERIC_RELOC_PB_LA_PTR",
        "GENERIC_RELOC_LOCAL_SECTDIFF",
        "GENERIC_RELOC_TLV" };

      if (RType > 5)
        res = "Unknown";
      else
        res = Table[RType];
      break;
    }
    case Triple::x86_64: {
      static const char *const Table[] =  {
        "X86_64_RELOC_UNSIGNED",
        "X86_64_RELOC_SIGNED",
        "X86_64_RELOC_BRANCH",
        "X86_64_RELOC_GOT_LOAD",
        "X86_64_RELOC_GOT",
        "X86_64_RELOC_SUBTRACTOR",
        "X86_64_RELOC_SIGNED_1",
        "X86_64_RELOC_SIGNED_2",
        "X86_64_RELOC_SIGNED_4",
        "X86_64_RELOC_TLV" };

      if (RType > 9)
        res = "Unknown";
      else
        res = Table[RType];
      break;
    }
    case Triple::arm: {
      static const char *const Table[] =  {
        "ARM_RELOC_VANILLA",
        "ARM_RELOC_PAIR",
        "ARM_RELOC_SECTDIFF",
        "ARM_RELOC_LOCAL_SECTDIFF",
        "ARM_RELOC_PB_LA_PTR",
        "ARM_RELOC_BR24",
        "ARM_THUMB_RELOC_BR22",
        "ARM_THUMB_32BIT_BRANCH",
        "ARM_RELOC_HALF",
        "ARM_RELOC_HALF_SECTDIFF" };

      if (RType > 9)
        res = "Unknown";
      else
        res = Table[RType];
      break;
    }
    case Triple::aarch64: {
      static const char *const Table[] = {
        "ARM64_RELOC_UNSIGNED",           "ARM64_RELOC_SUBTRACTOR",
        "ARM64_RELOC_BRANCH26",           "ARM64_RELOC_PAGE21",
        "ARM64_RELOC_PAGEOFF12",          "ARM64_RELOC_GOT_LOAD_PAGE21",
        "ARM64_RELOC_GOT_LOAD_PAGEOFF12", "ARM64_RELOC_POINTER_TO_GOT",
        "ARM64_RELOC_TLVP_LOAD_PAGE21",   "ARM64_RELOC_TLVP_LOAD_PAGEOFF12",
        "ARM64_RELOC_ADDEND"
      };

      if (RType >= array_lengthof(Table))
        res = "Unknown";
      else
        res = Table[RType];
      break;
    }
    case Triple::ppc: {
      static const char *const Table[] =  {
        "PPC_RELOC_VANILLA",
        "PPC_RELOC_PAIR",
        "PPC_RELOC_BR14",
        "PPC_RELOC_BR24",
        "PPC_RELOC_HI16",
        "PPC_RELOC_LO16",
        "PPC_RELOC_HA16",
        "PPC_RELOC_LO14",
        "PPC_RELOC_SECTDIFF",
        "PPC_RELOC_PB_LA_PTR",
        "PPC_RELOC_HI16_SECTDIFF",
        "PPC_RELOC_LO16_SECTDIFF",
        "PPC_RELOC_HA16_SECTDIFF",
        "PPC_RELOC_JBSR",
        "PPC_RELOC_LO14_SECTDIFF",
        "PPC_RELOC_LOCAL_SECTDIFF" };

      if (RType > 15)
        res = "Unknown";
      else
        res = Table[RType];
      break;
    }
    case Triple::UnknownArch:
      res = "Unknown";
      break;
  }
  Result.append(res.begin(), res.end());
  return object_error::success;
}

std::error_code
MachOObjectFile::getRelocationValueString(DataRefImpl Rel,
                                          SmallVectorImpl<char> &Result) const {
  MachO::any_relocation_info RE = getRelocation(Rel);

  unsigned Arch = this->getArch();

  std::string fmtbuf;
  raw_string_ostream fmt(fmtbuf);
  unsigned Type = this->getAnyRelocationType(RE);
  bool IsPCRel = this->getAnyRelocationPCRel(RE);

  // Determine any addends that should be displayed with the relocation.
  // These require decoding the relocation type, which is triple-specific.

  // X86_64 has entirely custom relocation types.
  if (Arch == Triple::x86_64) {
    bool isPCRel = getAnyRelocationPCRel(RE);

    switch (Type) {
      case MachO::X86_64_RELOC_GOT_LOAD:
      case MachO::X86_64_RELOC_GOT: {
        printRelocationTargetName(this, RE, fmt);
        fmt << "@GOT";
        if (isPCRel) fmt << "PCREL";
        break;
      }
      case MachO::X86_64_RELOC_SUBTRACTOR: {
        DataRefImpl RelNext = Rel;
        moveRelocationNext(RelNext);
        MachO::any_relocation_info RENext = getRelocation(RelNext);

        // X86_64_RELOC_SUBTRACTOR must be followed by a relocation of type
        // X86_64_RELOC_UNSIGNED.
        // NOTE: Scattered relocations don't exist on x86_64.
        unsigned RType = getAnyRelocationType(RENext);
        if (RType != MachO::X86_64_RELOC_UNSIGNED)
          report_fatal_error("Expected X86_64_RELOC_UNSIGNED after "
                             "X86_64_RELOC_SUBTRACTOR.");

        // The X86_64_RELOC_UNSIGNED contains the minuend symbol;
        // X86_64_RELOC_SUBTRACTOR contains the subtrahend.
        printRelocationTargetName(this, RENext, fmt);
        fmt << "-";
        printRelocationTargetName(this, RE, fmt);
        break;
      }
      case MachO::X86_64_RELOC_TLV:
        printRelocationTargetName(this, RE, fmt);
        fmt << "@TLV";
        if (isPCRel) fmt << "P";
        break;
      case MachO::X86_64_RELOC_SIGNED_1:
        printRelocationTargetName(this, RE, fmt);
        fmt << "-1";
        break;
      case MachO::X86_64_RELOC_SIGNED_2:
        printRelocationTargetName(this, RE, fmt);
        fmt << "-2";
        break;
      case MachO::X86_64_RELOC_SIGNED_4:
        printRelocationTargetName(this, RE, fmt);
        fmt << "-4";
        break;
      default:
        printRelocationTargetName(this, RE, fmt);
        break;
    }
  // X86 and ARM share some relocation types in common.
  } else if (Arch == Triple::x86 || Arch == Triple::arm ||
             Arch == Triple::ppc) {
    // Generic relocation types...
    switch (Type) {
      case MachO::GENERIC_RELOC_PAIR: // prints no info
        return object_error::success;
      case MachO::GENERIC_RELOC_SECTDIFF: {
        DataRefImpl RelNext = Rel;
        moveRelocationNext(RelNext);
        MachO::any_relocation_info RENext = getRelocation(RelNext);

        // X86 sect diff's must be followed by a relocation of type
        // GENERIC_RELOC_PAIR.
        unsigned RType = getAnyRelocationType(RENext);

        if (RType != MachO::GENERIC_RELOC_PAIR)
          report_fatal_error("Expected GENERIC_RELOC_PAIR after "
                             "GENERIC_RELOC_SECTDIFF.");

        printRelocationTargetName(this, RE, fmt);
        fmt << "-";
        printRelocationTargetName(this, RENext, fmt);
        break;
      }
    }

    if (Arch == Triple::x86 || Arch == Triple::ppc) {
      switch (Type) {
        case MachO::GENERIC_RELOC_LOCAL_SECTDIFF: {
          DataRefImpl RelNext = Rel;
          moveRelocationNext(RelNext);
          MachO::any_relocation_info RENext = getRelocation(RelNext);

          // X86 sect diff's must be followed by a relocation of type
          // GENERIC_RELOC_PAIR.
          unsigned RType = getAnyRelocationType(RENext);
          if (RType != MachO::GENERIC_RELOC_PAIR)
            report_fatal_error("Expected GENERIC_RELOC_PAIR after "
                               "GENERIC_RELOC_LOCAL_SECTDIFF.");

          printRelocationTargetName(this, RE, fmt);
          fmt << "-";
          printRelocationTargetName(this, RENext, fmt);
          break;
        }
        case MachO::GENERIC_RELOC_TLV: {
          printRelocationTargetName(this, RE, fmt);
          fmt << "@TLV";
          if (IsPCRel) fmt << "P";
          break;
        }
        default:
          printRelocationTargetName(this, RE, fmt);
      }
    } else { // ARM-specific relocations
      switch (Type) {
        case MachO::ARM_RELOC_HALF:
        case MachO::ARM_RELOC_HALF_SECTDIFF: {
          // Half relocations steal a bit from the length field to encode
          // whether this is an upper16 or a lower16 relocation.
          bool isUpper = getAnyRelocationLength(RE) >> 1;

          if (isUpper)
            fmt << ":upper16:(";
          else
            fmt << ":lower16:(";
          printRelocationTargetName(this, RE, fmt);

          DataRefImpl RelNext = Rel;
          moveRelocationNext(RelNext);
          MachO::any_relocation_info RENext = getRelocation(RelNext);

          // ARM half relocs must be followed by a relocation of type
          // ARM_RELOC_PAIR.
          unsigned RType = getAnyRelocationType(RENext);
          if (RType != MachO::ARM_RELOC_PAIR)
            report_fatal_error("Expected ARM_RELOC_PAIR after "
                               "ARM_RELOC_HALF");

          // NOTE: The half of the target virtual address is stashed in the
          // address field of the secondary relocation, but we can't reverse
          // engineer the constant offset from it without decoding the movw/movt
          // instruction to find the other half in its immediate field.

          // ARM_RELOC_HALF_SECTDIFF encodes the second section in the
          // symbol/section pointer of the follow-on relocation.
          if (Type == MachO::ARM_RELOC_HALF_SECTDIFF) {
            fmt << "-";
            printRelocationTargetName(this, RENext, fmt);
          }

          fmt << ")";
          break;
        }
        default: {
          printRelocationTargetName(this, RE, fmt);
        }
      }
    }
  } else
    printRelocationTargetName(this, RE, fmt);

  fmt.flush();
  Result.append(fmtbuf.begin(), fmtbuf.end());
  return object_error::success;
}

std::error_code MachOObjectFile::getRelocationHidden(DataRefImpl Rel,
                                                     bool &Result) const {
  unsigned Arch = getArch();
  uint64_t Type;
  getRelocationType(Rel, Type);

  Result = false;

  // On arches that use the generic relocations, GENERIC_RELOC_PAIR
  // is always hidden.
  if (Arch == Triple::x86 || Arch == Triple::arm || Arch == Triple::ppc) {
    if (Type == MachO::GENERIC_RELOC_PAIR) Result = true;
  } else if (Arch == Triple::x86_64) {
    // On x86_64, X86_64_RELOC_UNSIGNED is hidden only when it follows
    // an X86_64_RELOC_SUBTRACTOR.
    if (Type == MachO::X86_64_RELOC_UNSIGNED && Rel.d.a > 0) {
      DataRefImpl RelPrev = Rel;
      RelPrev.d.a--;
      uint64_t PrevType;
      getRelocationType(RelPrev, PrevType);
      if (PrevType == MachO::X86_64_RELOC_SUBTRACTOR)
        Result = true;
    }
  }

  return object_error::success;
}

//
// guessLibraryShortName() is passed a name of a dynamic library and returns a
// guess on what the short name is.  Then name is returned as a substring of the
// StringRef Name passed in.  The name of the dynamic library is recognized as
// a framework if it has one of the two following forms:
//      Foo.framework/Versions/A/Foo
//      Foo.framework/Foo
// Where A and Foo can be any string.  And may contain a trailing suffix
// starting with an underbar.  If the Name is recognized as a framework then
// isFramework is set to true else it is set to false.  If the Name has a
// suffix then Suffix is set to the substring in Name that contains the suffix
// else it is set to a NULL StringRef.
//
// The Name of the dynamic library is recognized as a library name if it has
// one of the two following forms:
//      libFoo.A.dylib
//      libFoo.dylib
// The library may have a suffix trailing the name Foo of the form:
//      libFoo_profile.A.dylib
//      libFoo_profile.dylib
//
// The Name of the dynamic library is also recognized as a library name if it
// has the following form:
//      Foo.qtx
//
// If the Name of the dynamic library is none of the forms above then a NULL
// StringRef is returned.
//
StringRef MachOObjectFile::guessLibraryShortName(StringRef Name,
                                                 bool &isFramework,
                                                 StringRef &Suffix) {
  StringRef Foo, F, DotFramework, V, Dylib, Lib, Dot, Qtx;
  size_t a, b, c, d, Idx;

  isFramework = false;
  Suffix = StringRef();

  // Pull off the last component and make Foo point to it
  a = Name.rfind('/');
  if (a == Name.npos || a == 0)
    goto guess_library;
  Foo = Name.slice(a+1, Name.npos);

  // Look for a suffix starting with a '_'
  Idx = Foo.rfind('_');
  if (Idx != Foo.npos && Foo.size() >= 2) {
    Suffix = Foo.slice(Idx, Foo.npos);
    Foo = Foo.slice(0, Idx);
  }

  // First look for the form Foo.framework/Foo
  b = Name.rfind('/', a);
  if (b == Name.npos)
    Idx = 0;
  else
    Idx = b+1;
  F = Name.slice(Idx, Idx + Foo.size());
  DotFramework = Name.slice(Idx + Foo.size(),
                            Idx + Foo.size() + sizeof(".framework/")-1);
  if (F == Foo && DotFramework == ".framework/") {
    isFramework = true;
    return Foo;
  }

  // Next look for the form Foo.framework/Versions/A/Foo
  if (b == Name.npos)
    goto guess_library;
  c =  Name.rfind('/', b);
  if (c == Name.npos || c == 0)
    goto guess_library;
  V = Name.slice(c+1, Name.npos);
  if (!V.startswith("Versions/"))
    goto guess_library;
  d =  Name.rfind('/', c);
  if (d == Name.npos)
    Idx = 0;
  else
    Idx = d+1;
  F = Name.slice(Idx, Idx + Foo.size());
  DotFramework = Name.slice(Idx + Foo.size(),
                            Idx + Foo.size() + sizeof(".framework/")-1);
  if (F == Foo && DotFramework == ".framework/") {
    isFramework = true;
    return Foo;
  }

guess_library:
  // pull off the suffix after the "." and make a point to it
  a = Name.rfind('.');
  if (a == Name.npos || a == 0)
    return StringRef();
  Dylib = Name.slice(a, Name.npos);
  if (Dylib != ".dylib")
    goto guess_qtx;

  // First pull off the version letter for the form Foo.A.dylib if any.
  if (a >= 3) {
    Dot = Name.slice(a-2, a-1);
    if (Dot == ".")
      a = a - 2;
  }

  b = Name.rfind('/', a);
  if (b == Name.npos)
    b = 0;
  else
    b = b+1;
  // ignore any suffix after an underbar like Foo_profile.A.dylib
  Idx = Name.find('_', b);
  if (Idx != Name.npos && Idx != b) {
    Lib = Name.slice(b, Idx);
    Suffix = Name.slice(Idx, a);
  }
  else
    Lib = Name.slice(b, a);
  // There are incorrect library names of the form:
  // libATS.A_profile.dylib so check for these.
  if (Lib.size() >= 3) {
    Dot = Lib.slice(Lib.size()-2, Lib.size()-1);
    if (Dot == ".")
      Lib = Lib.slice(0, Lib.size()-2);
  }
  return Lib;

guess_qtx:
  Qtx = Name.slice(a, Name.npos);
  if (Qtx != ".qtx")
    return StringRef();
  b = Name.rfind('/', a);
  if (b == Name.npos)
    Lib = Name.slice(0, a);
  else
    Lib = Name.slice(b+1, a);
  // There are library names of the form: QT.A.qtx so check for these.
  if (Lib.size() >= 3) {
    Dot = Lib.slice(Lib.size()-2, Lib.size()-1);
    if (Dot == ".")
      Lib = Lib.slice(0, Lib.size()-2);
  }
  return Lib;
}

// getLibraryShortNameByIndex() is used to get the short name of the library
// for an undefined symbol in a linked Mach-O binary that was linked with the
// normal two-level namespace default (that is MH_TWOLEVEL in the header).
// It is passed the index (0 - based) of the library as translated from
// GET_LIBRARY_ORDINAL (1 - based).
std::error_code MachOObjectFile::getLibraryShortNameByIndex(unsigned Index,
                                                         StringRef &Res) const {
  if (Index >= Libraries.size())
    return object_error::parse_failed;

  // If the cache of LibrariesShortNames is not built up do that first for
  // all the Libraries.
  if (LibrariesShortNames.size() == 0) {
    for (unsigned i = 0; i < Libraries.size(); i++) {
      MachO::dylib_command D =
        getStruct<MachO::dylib_command>(this, Libraries[i]);
      if (D.dylib.name >= D.cmdsize)
        return object_error::parse_failed;
      const char *P = (const char *)(Libraries[i]) + D.dylib.name;
      StringRef Name = StringRef(P);
      if (D.dylib.name+Name.size() >= D.cmdsize)
        return object_error::parse_failed;
      StringRef Suffix;
      bool isFramework;
      StringRef shortName = guessLibraryShortName(Name, isFramework, Suffix);
      if (shortName.empty())
        LibrariesShortNames.push_back(Name);
      else
        LibrariesShortNames.push_back(shortName);
    }
  }

  Res = LibrariesShortNames[Index];
  return object_error::success;
}

basic_symbol_iterator MachOObjectFile::symbol_begin_impl() const {
  return getSymbolByIndex(0);
}

basic_symbol_iterator MachOObjectFile::symbol_end_impl() const {
  DataRefImpl DRI;
  if (!SymtabLoadCmd)
    return basic_symbol_iterator(SymbolRef(DRI, this));

  MachO::symtab_command Symtab = getSymtabLoadCommand();
  unsigned SymbolTableEntrySize = is64Bit() ?
    sizeof(MachO::nlist_64) :
    sizeof(MachO::nlist);
  unsigned Offset = Symtab.symoff +
    Symtab.nsyms * SymbolTableEntrySize;
  DRI.p = reinterpret_cast<uintptr_t>(getPtr(this, Offset));
  return basic_symbol_iterator(SymbolRef(DRI, this));
}

basic_symbol_iterator MachOObjectFile::getSymbolByIndex(unsigned Index) const {
  DataRefImpl DRI;
  if (!SymtabLoadCmd)
    return basic_symbol_iterator(SymbolRef(DRI, this));

  MachO::symtab_command Symtab = getSymtabLoadCommand();
  assert(Index < Symtab.nsyms && "Requested symbol index is out of range.");
  unsigned SymbolTableEntrySize =
    is64Bit() ? sizeof(MachO::nlist_64) : sizeof(MachO::nlist);
  DRI.p = reinterpret_cast<uintptr_t>(getPtr(this, Symtab.symoff));
  DRI.p += Index * SymbolTableEntrySize;
  return basic_symbol_iterator(SymbolRef(DRI, this));
}

section_iterator MachOObjectFile::section_begin() const {
  DataRefImpl DRI;
  return section_iterator(SectionRef(DRI, this));
}

section_iterator MachOObjectFile::section_end() const {
  DataRefImpl DRI;
  DRI.d.a = Sections.size();
  return section_iterator(SectionRef(DRI, this));
}

uint8_t MachOObjectFile::getBytesInAddress() const {
  return is64Bit() ? 8 : 4;
}

StringRef MachOObjectFile::getFileFormatName() const {
  unsigned CPUType = getCPUType(this);
  if (!is64Bit()) {
    switch (CPUType) {
    case llvm::MachO::CPU_TYPE_I386:
      return "Mach-O 32-bit i386";
    case llvm::MachO::CPU_TYPE_ARM:
      return "Mach-O arm";
    case llvm::MachO::CPU_TYPE_POWERPC:
      return "Mach-O 32-bit ppc";
    default:
      assert((CPUType & llvm::MachO::CPU_ARCH_ABI64) == 0 &&
             "64-bit object file when we're not 64-bit?");
      return "Mach-O 32-bit unknown";
    }
  }

  // Make sure the cpu type has the correct mask.
  assert((CPUType & llvm::MachO::CPU_ARCH_ABI64)
         == llvm::MachO::CPU_ARCH_ABI64 &&
         "32-bit object file when we're 64-bit?");

  switch (CPUType) {
  case llvm::MachO::CPU_TYPE_X86_64:
    return "Mach-O 64-bit x86-64";
  case llvm::MachO::CPU_TYPE_ARM64:
    return "Mach-O arm64";
  case llvm::MachO::CPU_TYPE_POWERPC64:
    return "Mach-O 64-bit ppc64";
  default:
    return "Mach-O 64-bit unknown";
  }
}

Triple::ArchType MachOObjectFile::getArch(uint32_t CPUType) {
  switch (CPUType) {
  case llvm::MachO::CPU_TYPE_I386:
    return Triple::x86;
  case llvm::MachO::CPU_TYPE_X86_64:
    return Triple::x86_64;
  case llvm::MachO::CPU_TYPE_ARM:
    return Triple::arm;
  case llvm::MachO::CPU_TYPE_ARM64:
    return Triple::aarch64;
  case llvm::MachO::CPU_TYPE_POWERPC:
    return Triple::ppc;
  case llvm::MachO::CPU_TYPE_POWERPC64:
    return Triple::ppc64;
  default:
    return Triple::UnknownArch;
  }
}

Triple MachOObjectFile::getArch(uint32_t CPUType, uint32_t CPUSubType,
                                const char **McpuDefault) {
  if (McpuDefault)
    *McpuDefault = nullptr;

  switch (CPUType) {
  case MachO::CPU_TYPE_I386:
    switch (CPUSubType & ~MachO::CPU_SUBTYPE_MASK) {
    case MachO::CPU_SUBTYPE_I386_ALL:
      return Triple("i386-apple-darwin");
    default:
      return Triple();
    }
  case MachO::CPU_TYPE_X86_64:
    switch (CPUSubType & ~MachO::CPU_SUBTYPE_MASK) {
    case MachO::CPU_SUBTYPE_X86_64_ALL:
      return Triple("x86_64-apple-darwin");
    case MachO::CPU_SUBTYPE_X86_64_H:
      return Triple("x86_64h-apple-darwin");
    default:
      return Triple();
    }
  case MachO::CPU_TYPE_ARM:
    switch (CPUSubType & ~MachO::CPU_SUBTYPE_MASK) {
    case MachO::CPU_SUBTYPE_ARM_V4T:
      return Triple("armv4t-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_V5TEJ:
      return Triple("armv5e-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_XSCALE:
      return Triple("xscale-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_V6:
      return Triple("armv6-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_V6M:
      if (McpuDefault)
        *McpuDefault = "cortex-m0";
      return Triple("armv6m-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_V7:
      return Triple("armv7-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_V7EM:
      if (McpuDefault)
        *McpuDefault = "cortex-m4";
      return Triple("armv7em-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_V7K:
      return Triple("armv7k-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_V7M:
      if (McpuDefault)
        *McpuDefault = "cortex-m3";
      return Triple("armv7m-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_V7S:
      return Triple("armv7s-apple-darwin");
    default:
      return Triple();
    }
  case MachO::CPU_TYPE_ARM64:
    switch (CPUSubType & ~MachO::CPU_SUBTYPE_MASK) {
    case MachO::CPU_SUBTYPE_ARM64_ALL:
      return Triple("arm64-apple-darwin");
    default:
      return Triple();
    }
  case MachO::CPU_TYPE_POWERPC:
    switch (CPUSubType & ~MachO::CPU_SUBTYPE_MASK) {
    case MachO::CPU_SUBTYPE_POWERPC_ALL:
      return Triple("ppc-apple-darwin");
    default:
      return Triple();
    }
  case MachO::CPU_TYPE_POWERPC64:
    switch (CPUSubType & ~MachO::CPU_SUBTYPE_MASK) {
    case MachO::CPU_SUBTYPE_POWERPC_ALL:
      return Triple("ppc64-apple-darwin");
    default:
      return Triple();
    }
  default:
    return Triple();
  }
}

Triple MachOObjectFile::getThumbArch(uint32_t CPUType, uint32_t CPUSubType,
                                     const char **McpuDefault) {
  if (McpuDefault)
    *McpuDefault = nullptr;

  switch (CPUType) {
  case MachO::CPU_TYPE_ARM:
    switch (CPUSubType & ~MachO::CPU_SUBTYPE_MASK) {
    case MachO::CPU_SUBTYPE_ARM_V4T:
      return Triple("thumbv4t-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_V5TEJ:
      return Triple("thumbv5e-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_XSCALE:
      return Triple("xscale-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_V6:
      return Triple("thumbv6-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_V6M:
      if (McpuDefault)
        *McpuDefault = "cortex-m0";
      return Triple("thumbv6m-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_V7:
      return Triple("thumbv7-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_V7EM:
      if (McpuDefault)
        *McpuDefault = "cortex-m4";
      return Triple("thumbv7em-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_V7K:
      return Triple("thumbv7k-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_V7M:
      if (McpuDefault)
        *McpuDefault = "cortex-m3";
      return Triple("thumbv7m-apple-darwin");
    case MachO::CPU_SUBTYPE_ARM_V7S:
      return Triple("thumbv7s-apple-darwin");
    default:
      return Triple();
    }
  default:
    return Triple();
  }
}

Triple MachOObjectFile::getArch(uint32_t CPUType, uint32_t CPUSubType,
                                const char **McpuDefault,
				Triple *ThumbTriple) {
  Triple T = MachOObjectFile::getArch(CPUType, CPUSubType, McpuDefault);
  *ThumbTriple = MachOObjectFile::getThumbArch(CPUType, CPUSubType,
                                               McpuDefault);
  return T;
}

Triple MachOObjectFile::getHostArch() {
  return Triple(sys::getDefaultTargetTriple());
}

bool MachOObjectFile::isValidArch(StringRef ArchFlag) {
  return StringSwitch<bool>(ArchFlag)
      .Case("i386", true)
      .Case("x86_64", true)
      .Case("x86_64h", true)
      .Case("armv4t", true)
      .Case("arm", true)
      .Case("armv5e", true)
      .Case("armv6", true)
      .Case("armv6m", true)
      .Case("armv7em", true)
      .Case("armv7k", true)
      .Case("armv7m", true)
      .Case("armv7s", true)
      .Case("arm64", true)
      .Case("ppc", true)
      .Case("ppc64", true)
      .Default(false);
}

unsigned MachOObjectFile::getArch() const {
  return getArch(getCPUType(this));
}

Triple MachOObjectFile::getArch(const char **McpuDefault,
                                Triple *ThumbTriple) const {
  Triple T;
  if (is64Bit()) {
    MachO::mach_header_64 H_64;
    H_64 = getHeader64();
    T = MachOObjectFile::getArch(H_64.cputype, H_64.cpusubtype, McpuDefault);
    *ThumbTriple = MachOObjectFile::getThumbArch(H_64.cputype, H_64.cpusubtype,
                                                 McpuDefault);
  } else {
    MachO::mach_header H;
    H = getHeader();
    T = MachOObjectFile::getArch(H.cputype, H.cpusubtype, McpuDefault);
    *ThumbTriple = MachOObjectFile::getThumbArch(H.cputype, H.cpusubtype,
                                                 McpuDefault);
  }
  return T;
}

relocation_iterator MachOObjectFile::section_rel_begin(unsigned Index) const {
  DataRefImpl DRI;
  DRI.d.a = Index;
  return section_rel_begin(DRI);
}

relocation_iterator MachOObjectFile::section_rel_end(unsigned Index) const {
  DataRefImpl DRI;
  DRI.d.a = Index;
  return section_rel_end(DRI);
}

dice_iterator MachOObjectFile::begin_dices() const {
  DataRefImpl DRI;
  if (!DataInCodeLoadCmd)
    return dice_iterator(DiceRef(DRI, this));

  MachO::linkedit_data_command DicLC = getDataInCodeLoadCommand();
  DRI.p = reinterpret_cast<uintptr_t>(getPtr(this, DicLC.dataoff));
  return dice_iterator(DiceRef(DRI, this));
}

dice_iterator MachOObjectFile::end_dices() const {
  DataRefImpl DRI;
  if (!DataInCodeLoadCmd)
    return dice_iterator(DiceRef(DRI, this));

  MachO::linkedit_data_command DicLC = getDataInCodeLoadCommand();
  unsigned Offset = DicLC.dataoff + DicLC.datasize;
  DRI.p = reinterpret_cast<uintptr_t>(getPtr(this, Offset));
  return dice_iterator(DiceRef(DRI, this));
}

ExportEntry::ExportEntry(ArrayRef<uint8_t> T) 
  : Trie(T), Malformed(false), Done(false) { }

void ExportEntry::moveToFirst() {
  pushNode(0);
  pushDownUntilBottom();
}

void ExportEntry::moveToEnd() {
  Stack.clear();
  Done = true;
}

bool ExportEntry::operator==(const ExportEntry &Other) const {
  // Common case, one at end, other iterating from begin. 
  if (Done || Other.Done)
    return (Done == Other.Done);
  // Not equal if different stack sizes.
  if (Stack.size() != Other.Stack.size())
    return false;
  // Not equal if different cumulative strings.
  if (!CumulativeString.str().equals(Other.CumulativeString.str()))
    return false;
  // Equal if all nodes in both stacks match.
  for (unsigned i=0; i < Stack.size(); ++i) {
    if (Stack[i].Start != Other.Stack[i].Start)
      return false;
  }
  return true;  
}

uint64_t ExportEntry::readULEB128(const uint8_t *&Ptr) {
  unsigned Count;
  uint64_t Result = decodeULEB128(Ptr, &Count);
  Ptr += Count;
  if (Ptr > Trie.end()) {
    Ptr = Trie.end();
    Malformed = true;
  }
  return Result;
}

StringRef ExportEntry::name() const {
  return CumulativeString.str();
}

uint64_t ExportEntry::flags() const {
  return Stack.back().Flags;
}

uint64_t ExportEntry::address() const {
  return Stack.back().Address;
}

uint64_t ExportEntry::other() const {
  return Stack.back().Other;
}

StringRef ExportEntry::otherName() const {
  const char* ImportName = Stack.back().ImportName;
  if (ImportName)
    return StringRef(ImportName);
  return StringRef();
}

uint32_t ExportEntry::nodeOffset() const {
  return Stack.back().Start - Trie.begin();
}

ExportEntry::NodeState::NodeState(const uint8_t *Ptr) 
  : Start(Ptr), Current(Ptr), Flags(0), Address(0), Other(0), 
    ImportName(nullptr), ChildCount(0), NextChildIndex(0),  
    ParentStringLength(0), IsExportNode(false) {
}

void ExportEntry::pushNode(uint64_t offset) {
  const uint8_t *Ptr = Trie.begin() + offset;
  NodeState State(Ptr);
  uint64_t ExportInfoSize = readULEB128(State.Current);
  State.IsExportNode = (ExportInfoSize != 0);
  const uint8_t* Children = State.Current + ExportInfoSize;
  if (State.IsExportNode) {
    State.Flags = readULEB128(State.Current);
    if (State.Flags & MachO::EXPORT_SYMBOL_FLAGS_REEXPORT) {
      State.Address = 0;
      State.Other = readULEB128(State.Current); // dylib ordinal
      State.ImportName = reinterpret_cast<const char*>(State.Current);
    } else {
      State.Address = readULEB128(State.Current);
      if (State.Flags & MachO::EXPORT_SYMBOL_FLAGS_STUB_AND_RESOLVER)
        State.Other = readULEB128(State.Current); 
    }
  }
  State.ChildCount = *Children;
  State.Current = Children + 1;
  State.NextChildIndex = 0;
  State.ParentStringLength = CumulativeString.size();
  Stack.push_back(State);
}

void ExportEntry::pushDownUntilBottom() {
  while (Stack.back().NextChildIndex < Stack.back().ChildCount) {
    NodeState &Top = Stack.back();
    CumulativeString.resize(Top.ParentStringLength);
    for (;*Top.Current != 0; Top.Current++) {
      char C = *Top.Current;
      CumulativeString.push_back(C);
    }
    Top.Current += 1;
    uint64_t childNodeIndex = readULEB128(Top.Current);
    Top.NextChildIndex += 1;
    pushNode(childNodeIndex);
  }
  if (!Stack.back().IsExportNode) {
    Malformed = true;
    moveToEnd();
  }
}

// We have a trie data structure and need a way to walk it that is compatible
// with the C++ iterator model. The solution is a non-recursive depth first
// traversal where the iterator contains a stack of parent nodes along with a
// string that is the accumulation of all edge strings along the parent chain
// to this point.
//
// There is one "export" node for each exported symbol.  But because some
// symbols may be a prefix of another symbol (e.g. _dup and _dup2), an export
// node may have child nodes too.  
//
// The algorithm for moveNext() is to keep moving down the leftmost unvisited
// child until hitting a node with no children (which is an export node or
// else the trie is malformed). On the way down, each node is pushed on the
// stack ivar.  If there is no more ways down, it pops up one and tries to go
// down a sibling path until a childless node is reached.
void ExportEntry::moveNext() {
  if (Stack.empty() || !Stack.back().IsExportNode) {
    Malformed = true;
    moveToEnd();
    return;
  }

  Stack.pop_back();
  while (!Stack.empty()) {
    NodeState &Top = Stack.back();
    if (Top.NextChildIndex < Top.ChildCount) {
      pushDownUntilBottom();
      // Now at the next export node.
      return;
    } else {
      if (Top.IsExportNode) {
        // This node has no children but is itself an export node.
        CumulativeString.resize(Top.ParentStringLength);
        return;
      }
      Stack.pop_back();
    }
  }
  Done = true;
}

iterator_range<export_iterator> 
MachOObjectFile::exports(ArrayRef<uint8_t> Trie) {
  ExportEntry Start(Trie);
  Start.moveToFirst();

  ExportEntry Finish(Trie);
  Finish.moveToEnd();

  return iterator_range<export_iterator>(export_iterator(Start), 
                                         export_iterator(Finish));
}

iterator_range<export_iterator> MachOObjectFile::exports() const {
  return exports(getDyldInfoExportsTrie());
}


MachORebaseEntry::MachORebaseEntry(ArrayRef<uint8_t> Bytes, bool is64Bit)
    : Opcodes(Bytes), Ptr(Bytes.begin()), SegmentOffset(0), SegmentIndex(0),
      RemainingLoopCount(0), AdvanceAmount(0), RebaseType(0),
      PointerSize(is64Bit ? 8 : 4), Malformed(false), Done(false) {}

void MachORebaseEntry::moveToFirst() {
  Ptr = Opcodes.begin();
  moveNext();
}

void MachORebaseEntry::moveToEnd() {
  Ptr = Opcodes.end();
  RemainingLoopCount = 0;
  Done = true;
}

void MachORebaseEntry::moveNext() {
  // If in the middle of some loop, move to next rebasing in loop.
  SegmentOffset += AdvanceAmount;
  if (RemainingLoopCount) {
    --RemainingLoopCount;
    return;
  }
  if (Ptr == Opcodes.end()) {
    Done = true;
    return;
  }
  bool More = true;
  while (More && !Malformed) {
    // Parse next opcode and set up next loop.
    uint8_t Byte = *Ptr++;
    uint8_t ImmValue = Byte & MachO::REBASE_IMMEDIATE_MASK;
    uint8_t Opcode = Byte & MachO::REBASE_OPCODE_MASK;
    switch (Opcode) {
    case MachO::REBASE_OPCODE_DONE:
      More = false;
      Done = true;
      moveToEnd();
      DEBUG_WITH_TYPE("mach-o-rebase", llvm::dbgs() << "REBASE_OPCODE_DONE\n");
      break;
    case MachO::REBASE_OPCODE_SET_TYPE_IMM:
      RebaseType = ImmValue;
      DEBUG_WITH_TYPE(
          "mach-o-rebase",
          llvm::dbgs() << "REBASE_OPCODE_SET_TYPE_IMM: "
                       << "RebaseType=" << (int) RebaseType << "\n");
      break;
    case MachO::REBASE_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB:
      SegmentIndex = ImmValue;
      SegmentOffset = readULEB128();
      DEBUG_WITH_TYPE(
          "mach-o-rebase",
          llvm::dbgs() << "REBASE_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB: "
                       << "SegmentIndex=" << SegmentIndex << ", "
                       << format("SegmentOffset=0x%06X", SegmentOffset)
                       << "\n");
      break;
    case MachO::REBASE_OPCODE_ADD_ADDR_ULEB:
      SegmentOffset += readULEB128();
      DEBUG_WITH_TYPE("mach-o-rebase",
                      llvm::dbgs() << "REBASE_OPCODE_ADD_ADDR_ULEB: "
                                   << format("SegmentOffset=0x%06X",
                                             SegmentOffset) << "\n");
      break;
    case MachO::REBASE_OPCODE_ADD_ADDR_IMM_SCALED:
      SegmentOffset += ImmValue * PointerSize;
      DEBUG_WITH_TYPE("mach-o-rebase",
                      llvm::dbgs() << "REBASE_OPCODE_ADD_ADDR_IMM_SCALED: "
                                   << format("SegmentOffset=0x%06X",
                                             SegmentOffset) << "\n");
      break;
    case MachO::REBASE_OPCODE_DO_REBASE_IMM_TIMES:
      AdvanceAmount = PointerSize;
      RemainingLoopCount = ImmValue - 1;
      DEBUG_WITH_TYPE(
          "mach-o-rebase",
          llvm::dbgs() << "REBASE_OPCODE_DO_REBASE_IMM_TIMES: "
                       << format("SegmentOffset=0x%06X", SegmentOffset)
                       << ", AdvanceAmount=" << AdvanceAmount
                       << ", RemainingLoopCount=" << RemainingLoopCount
                       << "\n");
      return;
    case MachO::REBASE_OPCODE_DO_REBASE_ULEB_TIMES:
      AdvanceAmount = PointerSize;
      RemainingLoopCount = readULEB128() - 1;
      DEBUG_WITH_TYPE(
          "mach-o-rebase",
          llvm::dbgs() << "REBASE_OPCODE_DO_REBASE_ULEB_TIMES: "
                       << format("SegmentOffset=0x%06X", SegmentOffset)
                       << ", AdvanceAmount=" << AdvanceAmount
                       << ", RemainingLoopCount=" << RemainingLoopCount
                       << "\n");
      return;
    case MachO::REBASE_OPCODE_DO_REBASE_ADD_ADDR_ULEB:
      AdvanceAmount = readULEB128() + PointerSize;
      RemainingLoopCount = 0;
      DEBUG_WITH_TYPE(
          "mach-o-rebase",
          llvm::dbgs() << "REBASE_OPCODE_DO_REBASE_ADD_ADDR_ULEB: "
                       << format("SegmentOffset=0x%06X", SegmentOffset)
                       << ", AdvanceAmount=" << AdvanceAmount
                       << ", RemainingLoopCount=" << RemainingLoopCount
                       << "\n");
      return;
    case MachO::REBASE_OPCODE_DO_REBASE_ULEB_TIMES_SKIPPING_ULEB:
      RemainingLoopCount = readULEB128() - 1;
      AdvanceAmount = readULEB128() + PointerSize;
      DEBUG_WITH_TYPE(
          "mach-o-rebase",
          llvm::dbgs() << "REBASE_OPCODE_DO_REBASE_ULEB_TIMES_SKIPPING_ULEB: "
                       << format("SegmentOffset=0x%06X", SegmentOffset)
                       << ", AdvanceAmount=" << AdvanceAmount
                       << ", RemainingLoopCount=" << RemainingLoopCount
                       << "\n");
      return;
    default:
      Malformed = true;
    }
  }
}

uint64_t MachORebaseEntry::readULEB128() {
  unsigned Count;
  uint64_t Result = decodeULEB128(Ptr, &Count);
  Ptr += Count;
  if (Ptr > Opcodes.end()) {
    Ptr = Opcodes.end();
    Malformed = true;
  }
  return Result;
}

uint32_t MachORebaseEntry::segmentIndex() const { return SegmentIndex; }

uint64_t MachORebaseEntry::segmentOffset() const { return SegmentOffset; }

StringRef MachORebaseEntry::typeName() const {
  switch (RebaseType) {
  case MachO::REBASE_TYPE_POINTER:
    return "pointer";
  case MachO::REBASE_TYPE_TEXT_ABSOLUTE32:
    return "text abs32";
  case MachO::REBASE_TYPE_TEXT_PCREL32:
    return "text rel32";
  }
  return "unknown";
}

bool MachORebaseEntry::operator==(const MachORebaseEntry &Other) const {
  assert(Opcodes == Other.Opcodes && "compare iterators of different files");
  return (Ptr == Other.Ptr) &&
         (RemainingLoopCount == Other.RemainingLoopCount) &&
         (Done == Other.Done);
}

iterator_range<rebase_iterator>
MachOObjectFile::rebaseTable(ArrayRef<uint8_t> Opcodes, bool is64) {
  MachORebaseEntry Start(Opcodes, is64);
  Start.moveToFirst();

  MachORebaseEntry Finish(Opcodes, is64);
  Finish.moveToEnd();

  return iterator_range<rebase_iterator>(rebase_iterator(Start),
                                         rebase_iterator(Finish));
}

iterator_range<rebase_iterator> MachOObjectFile::rebaseTable() const {
  return rebaseTable(getDyldInfoRebaseOpcodes(), is64Bit());
}


MachOBindEntry::MachOBindEntry(ArrayRef<uint8_t> Bytes, bool is64Bit,
                               Kind BK)
    : Opcodes(Bytes), Ptr(Bytes.begin()), SegmentOffset(0), SegmentIndex(0),
      Ordinal(0), Flags(0), Addend(0), RemainingLoopCount(0), AdvanceAmount(0),
      BindType(0), PointerSize(is64Bit ? 8 : 4),
      TableKind(BK), Malformed(false), Done(false) {}

void MachOBindEntry::moveToFirst() {
  Ptr = Opcodes.begin();
  moveNext();
}

void MachOBindEntry::moveToEnd() {
  Ptr = Opcodes.end();
  RemainingLoopCount = 0;
  Done = true;
}

void MachOBindEntry::moveNext() {
  // If in the middle of some loop, move to next binding in loop.
  SegmentOffset += AdvanceAmount;
  if (RemainingLoopCount) {
    --RemainingLoopCount;
    return;
  }
  if (Ptr == Opcodes.end()) {
    Done = true;
    return;
  }
  bool More = true;
  while (More && !Malformed) {
    // Parse next opcode and set up next loop.
    uint8_t Byte = *Ptr++;
    uint8_t ImmValue = Byte & MachO::BIND_IMMEDIATE_MASK;
    uint8_t Opcode = Byte & MachO::BIND_OPCODE_MASK;
    int8_t SignExtended;
    const uint8_t *SymStart;
    switch (Opcode) {
    case MachO::BIND_OPCODE_DONE:
      if (TableKind == Kind::Lazy) {
        // Lazying bindings have a DONE opcode between entries.  Need to ignore
        // it to advance to next entry.  But need not if this is last entry.
        bool NotLastEntry = false;
        for (const uint8_t *P = Ptr; P < Opcodes.end(); ++P) {
          if (*P) {
            NotLastEntry = true;
          }
        }
        if (NotLastEntry)
          break;
      }
      More = false;
      Done = true;
      moveToEnd();
      DEBUG_WITH_TYPE("mach-o-bind", llvm::dbgs() << "BIND_OPCODE_DONE\n");
      break;
    case MachO::BIND_OPCODE_SET_DYLIB_ORDINAL_IMM:
      Ordinal = ImmValue;
      DEBUG_WITH_TYPE(
          "mach-o-bind",
          llvm::dbgs() << "BIND_OPCODE_SET_DYLIB_ORDINAL_IMM: "
                       << "Ordinal=" << Ordinal << "\n");
      break;
    case MachO::BIND_OPCODE_SET_DYLIB_ORDINAL_ULEB:
      Ordinal = readULEB128();
      DEBUG_WITH_TYPE(
          "mach-o-bind",
          llvm::dbgs() << "BIND_OPCODE_SET_DYLIB_ORDINAL_ULEB: "
                       << "Ordinal=" << Ordinal << "\n");
      break;
    case MachO::BIND_OPCODE_SET_DYLIB_SPECIAL_IMM:
      if (ImmValue) {
        SignExtended = MachO::BIND_OPCODE_MASK | ImmValue;
        Ordinal = SignExtended;
      } else
        Ordinal = 0;
      DEBUG_WITH_TYPE(
          "mach-o-bind",
          llvm::dbgs() << "BIND_OPCODE_SET_DYLIB_SPECIAL_IMM: "
                       << "Ordinal=" << Ordinal << "\n");
      break;
    case MachO::BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM:
      Flags = ImmValue;
      SymStart = Ptr;
      while (*Ptr) {
        ++Ptr;
      }
      SymbolName = StringRef(reinterpret_cast<const char*>(SymStart),
                             Ptr-SymStart);
      ++Ptr;
      DEBUG_WITH_TYPE(
          "mach-o-bind",
          llvm::dbgs() << "BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM: "
                       << "SymbolName=" << SymbolName << "\n");
      if (TableKind == Kind::Weak) {
        if (ImmValue & MachO::BIND_SYMBOL_FLAGS_NON_WEAK_DEFINITION)
          return;
      }
      break;
    case MachO::BIND_OPCODE_SET_TYPE_IMM:
      BindType = ImmValue;
      DEBUG_WITH_TYPE(
          "mach-o-bind",
          llvm::dbgs() << "BIND_OPCODE_SET_TYPE_IMM: "
                       << "BindType=" << (int)BindType << "\n");
      break;
    case MachO::BIND_OPCODE_SET_ADDEND_SLEB:
      Addend = readSLEB128();
      if (TableKind == Kind::Lazy)
        Malformed = true;
      DEBUG_WITH_TYPE(
          "mach-o-bind",
          llvm::dbgs() << "BIND_OPCODE_SET_ADDEND_SLEB: "
                       << "Addend=" << Addend << "\n");
      break;
    case MachO::BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB:
      SegmentIndex = ImmValue;
      SegmentOffset = readULEB128();
      DEBUG_WITH_TYPE(
          "mach-o-bind",
          llvm::dbgs() << "BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB: "
                       << "SegmentIndex=" << SegmentIndex << ", "
                       << format("SegmentOffset=0x%06X", SegmentOffset)
                       << "\n");
      break;
    case MachO::BIND_OPCODE_ADD_ADDR_ULEB:
      SegmentOffset += readULEB128();
      DEBUG_WITH_TYPE("mach-o-bind",
                      llvm::dbgs() << "BIND_OPCODE_ADD_ADDR_ULEB: "
                                   << format("SegmentOffset=0x%06X",
                                             SegmentOffset) << "\n");
      break;
    case MachO::BIND_OPCODE_DO_BIND:
      AdvanceAmount = PointerSize;
      RemainingLoopCount = 0;
      DEBUG_WITH_TYPE("mach-o-bind",
                      llvm::dbgs() << "BIND_OPCODE_DO_BIND: "
                                   << format("SegmentOffset=0x%06X",
                                             SegmentOffset) << "\n");
      return;
     case MachO::BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB:
      AdvanceAmount = readULEB128() + PointerSize;
      RemainingLoopCount = 0;
      if (TableKind == Kind::Lazy)
        Malformed = true;
      DEBUG_WITH_TYPE(
          "mach-o-bind",
          llvm::dbgs() << "BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB: "
                       << format("SegmentOffset=0x%06X", SegmentOffset)
                       << ", AdvanceAmount=" << AdvanceAmount
                       << ", RemainingLoopCount=" << RemainingLoopCount
                       << "\n");
      return;
    case MachO::BIND_OPCODE_DO_BIND_ADD_ADDR_IMM_SCALED:
      AdvanceAmount = ImmValue * PointerSize + PointerSize;
      RemainingLoopCount = 0;
      if (TableKind == Kind::Lazy)
        Malformed = true;
      DEBUG_WITH_TYPE("mach-o-bind",
                      llvm::dbgs()
                      << "BIND_OPCODE_DO_BIND_ADD_ADDR_IMM_SCALED: "
                      << format("SegmentOffset=0x%06X",
                                             SegmentOffset) << "\n");
      return;
    case MachO::BIND_OPCODE_DO_BIND_ULEB_TIMES_SKIPPING_ULEB:
      RemainingLoopCount = readULEB128() - 1;
      AdvanceAmount = readULEB128() + PointerSize;
      if (TableKind == Kind::Lazy)
        Malformed = true;
      DEBUG_WITH_TYPE(
          "mach-o-bind",
          llvm::dbgs() << "BIND_OPCODE_DO_BIND_ULEB_TIMES_SKIPPING_ULEB: "
                       << format("SegmentOffset=0x%06X", SegmentOffset)
                       << ", AdvanceAmount=" << AdvanceAmount
                       << ", RemainingLoopCount=" << RemainingLoopCount
                       << "\n");
      return;
    default:
      Malformed = true;
    }
  }
}

uint64_t MachOBindEntry::readULEB128() {
  unsigned Count;
  uint64_t Result = decodeULEB128(Ptr, &Count);
  Ptr += Count;
  if (Ptr > Opcodes.end()) {
    Ptr = Opcodes.end();
    Malformed = true;
  }
  return Result;
}

int64_t MachOBindEntry::readSLEB128() {
  unsigned Count;
  int64_t Result = decodeSLEB128(Ptr, &Count);
  Ptr += Count;
  if (Ptr > Opcodes.end()) {
    Ptr = Opcodes.end();
    Malformed = true;
  }
  return Result;
}


uint32_t MachOBindEntry::segmentIndex() const { return SegmentIndex; }

uint64_t MachOBindEntry::segmentOffset() const { return SegmentOffset; }

StringRef MachOBindEntry::typeName() const {
  switch (BindType) {
  case MachO::BIND_TYPE_POINTER:
    return "pointer";
  case MachO::BIND_TYPE_TEXT_ABSOLUTE32:
    return "text abs32";
  case MachO::BIND_TYPE_TEXT_PCREL32:
    return "text rel32";
  }
  return "unknown";
}

StringRef MachOBindEntry::symbolName() const { return SymbolName; }

int64_t MachOBindEntry::addend() const { return Addend; }

uint32_t MachOBindEntry::flags() const { return Flags; }

int MachOBindEntry::ordinal() const { return Ordinal; }

bool MachOBindEntry::operator==(const MachOBindEntry &Other) const {
  assert(Opcodes == Other.Opcodes && "compare iterators of different files");
  return (Ptr == Other.Ptr) &&
         (RemainingLoopCount == Other.RemainingLoopCount) &&
         (Done == Other.Done);
}

iterator_range<bind_iterator>
MachOObjectFile::bindTable(ArrayRef<uint8_t> Opcodes, bool is64,
                           MachOBindEntry::Kind BKind) {
  MachOBindEntry Start(Opcodes, is64, BKind);
  Start.moveToFirst();

  MachOBindEntry Finish(Opcodes, is64, BKind);
  Finish.moveToEnd();

  return iterator_range<bind_iterator>(bind_iterator(Start),
                                       bind_iterator(Finish));
}

iterator_range<bind_iterator> MachOObjectFile::bindTable() const {
  return bindTable(getDyldInfoBindOpcodes(), is64Bit(),
                   MachOBindEntry::Kind::Regular);
}

iterator_range<bind_iterator> MachOObjectFile::lazyBindTable() const {
  return bindTable(getDyldInfoLazyBindOpcodes(), is64Bit(),
                   MachOBindEntry::Kind::Lazy);
}

iterator_range<bind_iterator> MachOObjectFile::weakBindTable() const {
  return bindTable(getDyldInfoWeakBindOpcodes(), is64Bit(),
                   MachOBindEntry::Kind::Weak);
}

StringRef
MachOObjectFile::getSectionFinalSegmentName(DataRefImpl Sec) const {
  ArrayRef<char> Raw = getSectionRawFinalSegmentName(Sec);
  return parseSegmentOrSectionName(Raw.data());
}

ArrayRef<char>
MachOObjectFile::getSectionRawName(DataRefImpl Sec) const {
  const section_base *Base =
    reinterpret_cast<const section_base *>(Sections[Sec.d.a]);
  return makeArrayRef(Base->sectname);
}

ArrayRef<char>
MachOObjectFile::getSectionRawFinalSegmentName(DataRefImpl Sec) const {
  const section_base *Base =
    reinterpret_cast<const section_base *>(Sections[Sec.d.a]);
  return makeArrayRef(Base->segname);
}

bool
MachOObjectFile::isRelocationScattered(const MachO::any_relocation_info &RE)
  const {
  if (getCPUType(this) == MachO::CPU_TYPE_X86_64)
    return false;
  return getPlainRelocationAddress(RE) & MachO::R_SCATTERED;
}

unsigned MachOObjectFile::getPlainRelocationSymbolNum(
    const MachO::any_relocation_info &RE) const {
  if (isLittleEndian())
    return RE.r_word1 & 0xffffff;
  return RE.r_word1 >> 8;
}

bool MachOObjectFile::getPlainRelocationExternal(
    const MachO::any_relocation_info &RE) const {
  if (isLittleEndian())
    return (RE.r_word1 >> 27) & 1;
  return (RE.r_word1 >> 4) & 1;
}

bool MachOObjectFile::getScatteredRelocationScattered(
    const MachO::any_relocation_info &RE) const {
  return RE.r_word0 >> 31;
}

uint32_t MachOObjectFile::getScatteredRelocationValue(
    const MachO::any_relocation_info &RE) const {
  return RE.r_word1;
}

unsigned MachOObjectFile::getAnyRelocationAddress(
    const MachO::any_relocation_info &RE) const {
  if (isRelocationScattered(RE))
    return getScatteredRelocationAddress(RE);
  return getPlainRelocationAddress(RE);
}

unsigned MachOObjectFile::getAnyRelocationPCRel(
    const MachO::any_relocation_info &RE) const {
  if (isRelocationScattered(RE))
    return getScatteredRelocationPCRel(this, RE);
  return getPlainRelocationPCRel(this, RE);
}

unsigned MachOObjectFile::getAnyRelocationLength(
    const MachO::any_relocation_info &RE) const {
  if (isRelocationScattered(RE))
    return getScatteredRelocationLength(RE);
  return getPlainRelocationLength(this, RE);
}

unsigned
MachOObjectFile::getAnyRelocationType(
                                   const MachO::any_relocation_info &RE) const {
  if (isRelocationScattered(RE))
    return getScatteredRelocationType(RE);
  return getPlainRelocationType(this, RE);
}

SectionRef
MachOObjectFile::getRelocationSection(
                                   const MachO::any_relocation_info &RE) const {
  if (isRelocationScattered(RE) || getPlainRelocationExternal(RE))
    return *section_end();
  unsigned SecNum = getPlainRelocationSymbolNum(RE) - 1;
  DataRefImpl DRI;
  DRI.d.a = SecNum;
  return SectionRef(DRI, this);
}

MachOObjectFile::LoadCommandInfo
MachOObjectFile::getFirstLoadCommandInfo() const {
  MachOObjectFile::LoadCommandInfo Load;

  unsigned HeaderSize = is64Bit() ? sizeof(MachO::mach_header_64) :
                                    sizeof(MachO::mach_header);
  Load.Ptr = getPtr(this, HeaderSize);
  Load.C = getStruct<MachO::load_command>(this, Load.Ptr);
  return Load;
}

MachOObjectFile::LoadCommandInfo
MachOObjectFile::getNextLoadCommandInfo(const LoadCommandInfo &L) const {
  MachOObjectFile::LoadCommandInfo Next;
  Next.Ptr = L.Ptr + L.C.cmdsize;
  Next.C = getStruct<MachO::load_command>(this, Next.Ptr);
  return Next;
}

MachO::section MachOObjectFile::getSection(DataRefImpl DRI) const {
  return getStruct<MachO::section>(this, Sections[DRI.d.a]);
}

MachO::section_64 MachOObjectFile::getSection64(DataRefImpl DRI) const {
  return getStruct<MachO::section_64>(this, Sections[DRI.d.a]);
}

MachO::section MachOObjectFile::getSection(const LoadCommandInfo &L,
                                           unsigned Index) const {
  const char *Sec = getSectionPtr(this, L, Index);
  return getStruct<MachO::section>(this, Sec);
}

MachO::section_64 MachOObjectFile::getSection64(const LoadCommandInfo &L,
                                                unsigned Index) const {
  const char *Sec = getSectionPtr(this, L, Index);
  return getStruct<MachO::section_64>(this, Sec);
}

MachO::nlist
MachOObjectFile::getSymbolTableEntry(DataRefImpl DRI) const {
  const char *P = reinterpret_cast<const char *>(DRI.p);
  return getStruct<MachO::nlist>(this, P);
}

MachO::nlist_64
MachOObjectFile::getSymbol64TableEntry(DataRefImpl DRI) const {
  const char *P = reinterpret_cast<const char *>(DRI.p);
  return getStruct<MachO::nlist_64>(this, P);
}

MachO::linkedit_data_command
MachOObjectFile::getLinkeditDataLoadCommand(const LoadCommandInfo &L) const {
  return getStruct<MachO::linkedit_data_command>(this, L.Ptr);
}

MachO::segment_command
MachOObjectFile::getSegmentLoadCommand(const LoadCommandInfo &L) const {
  return getStruct<MachO::segment_command>(this, L.Ptr);
}

MachO::segment_command_64
MachOObjectFile::getSegment64LoadCommand(const LoadCommandInfo &L) const {
  return getStruct<MachO::segment_command_64>(this, L.Ptr);
}

MachO::linker_options_command
MachOObjectFile::getLinkerOptionsLoadCommand(const LoadCommandInfo &L) const {
  return getStruct<MachO::linker_options_command>(this, L.Ptr);
}

MachO::version_min_command
MachOObjectFile::getVersionMinLoadCommand(const LoadCommandInfo &L) const {
  return getStruct<MachO::version_min_command>(this, L.Ptr);
}

MachO::dylib_command
MachOObjectFile::getDylibIDLoadCommand(const LoadCommandInfo &L) const {
  return getStruct<MachO::dylib_command>(this, L.Ptr);
}

MachO::dyld_info_command
MachOObjectFile::getDyldInfoLoadCommand(const LoadCommandInfo &L) const {
  return getStruct<MachO::dyld_info_command>(this, L.Ptr);
}

MachO::dylinker_command
MachOObjectFile::getDylinkerCommand(const LoadCommandInfo &L) const {
  return getStruct<MachO::dylinker_command>(this, L.Ptr);
}

MachO::uuid_command
MachOObjectFile::getUuidCommand(const LoadCommandInfo &L) const {
  return getStruct<MachO::uuid_command>(this, L.Ptr);
}

MachO::source_version_command
MachOObjectFile::getSourceVersionCommand(const LoadCommandInfo &L) const {
  return getStruct<MachO::source_version_command>(this, L.Ptr);
}

MachO::entry_point_command
MachOObjectFile::getEntryPointCommand(const LoadCommandInfo &L) const {
  return getStruct<MachO::entry_point_command>(this, L.Ptr);
}


MachO::any_relocation_info
MachOObjectFile::getRelocation(DataRefImpl Rel) const {
  DataRefImpl Sec;
  Sec.d.a = Rel.d.a;
  uint32_t Offset;
  if (is64Bit()) {
    MachO::section_64 Sect = getSection64(Sec);
    Offset = Sect.reloff;
  } else {
    MachO::section Sect = getSection(Sec);
    Offset = Sect.reloff;
  }

  auto P = reinterpret_cast<const MachO::any_relocation_info *>(
      getPtr(this, Offset)) + Rel.d.b;
  return getStruct<MachO::any_relocation_info>(
      this, reinterpret_cast<const char *>(P));
}

MachO::data_in_code_entry
MachOObjectFile::getDice(DataRefImpl Rel) const {
  const char *P = reinterpret_cast<const char *>(Rel.p);
  return getStruct<MachO::data_in_code_entry>(this, P);
}

MachO::mach_header MachOObjectFile::getHeader() const {
  return getStruct<MachO::mach_header>(this, getPtr(this, 0));
}

MachO::mach_header_64 MachOObjectFile::getHeader64() const {
  return getStruct<MachO::mach_header_64>(this, getPtr(this, 0));
}

uint32_t MachOObjectFile::getIndirectSymbolTableEntry(
                                             const MachO::dysymtab_command &DLC,
                                             unsigned Index) const {
  uint64_t Offset = DLC.indirectsymoff + Index * sizeof(uint32_t);
  return getStruct<uint32_t>(this, getPtr(this, Offset));
}

MachO::data_in_code_entry
MachOObjectFile::getDataInCodeTableEntry(uint32_t DataOffset,
                                         unsigned Index) const {
  uint64_t Offset = DataOffset + Index * sizeof(MachO::data_in_code_entry);
  return getStruct<MachO::data_in_code_entry>(this, getPtr(this, Offset));
}

MachO::symtab_command MachOObjectFile::getSymtabLoadCommand() const {
  if (SymtabLoadCmd)
    return getStruct<MachO::symtab_command>(this, SymtabLoadCmd);

  // If there is no SymtabLoadCmd return a load command with zero'ed fields.
  MachO::symtab_command Cmd;
  Cmd.cmd = MachO::LC_SYMTAB;
  Cmd.cmdsize = sizeof(MachO::symtab_command);
  Cmd.symoff = 0;
  Cmd.nsyms = 0;
  Cmd.stroff = 0;
  Cmd.strsize = 0;
  return Cmd;
}

MachO::dysymtab_command MachOObjectFile::getDysymtabLoadCommand() const {
  if (DysymtabLoadCmd)
    return getStruct<MachO::dysymtab_command>(this, DysymtabLoadCmd);

  // If there is no DysymtabLoadCmd return a load command with zero'ed fields.
  MachO::dysymtab_command Cmd;
  Cmd.cmd = MachO::LC_DYSYMTAB;
  Cmd.cmdsize = sizeof(MachO::dysymtab_command);
  Cmd.ilocalsym = 0;
  Cmd.nlocalsym = 0;
  Cmd.iextdefsym = 0;
  Cmd.nextdefsym = 0;
  Cmd.iundefsym = 0;
  Cmd.nundefsym = 0;
  Cmd.tocoff = 0;
  Cmd.ntoc = 0;
  Cmd.modtaboff = 0;
  Cmd.nmodtab = 0;
  Cmd.extrefsymoff = 0;
  Cmd.nextrefsyms = 0;
  Cmd.indirectsymoff = 0;
  Cmd.nindirectsyms = 0;
  Cmd.extreloff = 0;
  Cmd.nextrel = 0;
  Cmd.locreloff = 0;
  Cmd.nlocrel = 0;
  return Cmd;
}

MachO::linkedit_data_command
MachOObjectFile::getDataInCodeLoadCommand() const {
  if (DataInCodeLoadCmd)
    return getStruct<MachO::linkedit_data_command>(this, DataInCodeLoadCmd);

  // If there is no DataInCodeLoadCmd return a load command with zero'ed fields.
  MachO::linkedit_data_command Cmd;
  Cmd.cmd = MachO::LC_DATA_IN_CODE;
  Cmd.cmdsize = sizeof(MachO::linkedit_data_command);
  Cmd.dataoff = 0;
  Cmd.datasize = 0;
  return Cmd;
}

ArrayRef<uint8_t> MachOObjectFile::getDyldInfoRebaseOpcodes() const {
  if (!DyldInfoLoadCmd) 
    return ArrayRef<uint8_t>();

  MachO::dyld_info_command DyldInfo 
                   = getStruct<MachO::dyld_info_command>(this, DyldInfoLoadCmd);
  const uint8_t *Ptr = reinterpret_cast<const uint8_t*>(
                                             getPtr(this, DyldInfo.rebase_off));
  return ArrayRef<uint8_t>(Ptr, DyldInfo.rebase_size);
}

ArrayRef<uint8_t> MachOObjectFile::getDyldInfoBindOpcodes() const {
  if (!DyldInfoLoadCmd) 
    return ArrayRef<uint8_t>();

  MachO::dyld_info_command DyldInfo 
                   = getStruct<MachO::dyld_info_command>(this, DyldInfoLoadCmd);
  const uint8_t *Ptr = reinterpret_cast<const uint8_t*>(
                                               getPtr(this, DyldInfo.bind_off));
  return ArrayRef<uint8_t>(Ptr, DyldInfo.bind_size);
}

ArrayRef<uint8_t> MachOObjectFile::getDyldInfoWeakBindOpcodes() const {
  if (!DyldInfoLoadCmd) 
    return ArrayRef<uint8_t>();

  MachO::dyld_info_command DyldInfo 
                   = getStruct<MachO::dyld_info_command>(this, DyldInfoLoadCmd);
  const uint8_t *Ptr = reinterpret_cast<const uint8_t*>(
                                          getPtr(this, DyldInfo.weak_bind_off));
  return ArrayRef<uint8_t>(Ptr, DyldInfo.weak_bind_size);
}

ArrayRef<uint8_t> MachOObjectFile::getDyldInfoLazyBindOpcodes() const {
  if (!DyldInfoLoadCmd) 
    return ArrayRef<uint8_t>();

  MachO::dyld_info_command DyldInfo 
                   = getStruct<MachO::dyld_info_command>(this, DyldInfoLoadCmd);
  const uint8_t *Ptr = reinterpret_cast<const uint8_t*>(
                                          getPtr(this, DyldInfo.lazy_bind_off));
  return ArrayRef<uint8_t>(Ptr, DyldInfo.lazy_bind_size);
}

ArrayRef<uint8_t> MachOObjectFile::getDyldInfoExportsTrie() const {
  if (!DyldInfoLoadCmd) 
    return ArrayRef<uint8_t>();

  MachO::dyld_info_command DyldInfo 
                   = getStruct<MachO::dyld_info_command>(this, DyldInfoLoadCmd);
  const uint8_t *Ptr = reinterpret_cast<const uint8_t*>(
                                             getPtr(this, DyldInfo.export_off));
  return ArrayRef<uint8_t>(Ptr, DyldInfo.export_size);
}

ArrayRef<uint8_t> MachOObjectFile::getUuid() const {
  if (!UuidLoadCmd)
    return ArrayRef<uint8_t>();
  // Returning a pointer is fine as uuid doesn't need endian swapping.
  const char *Ptr = UuidLoadCmd + offsetof(MachO::uuid_command, uuid);
  return ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(Ptr), 16);
}

StringRef MachOObjectFile::getStringTableData() const {
  MachO::symtab_command S = getSymtabLoadCommand();
  return getData().substr(S.stroff, S.strsize);
}

bool MachOObjectFile::is64Bit() const {
  return getType() == getMachOType(false, true) ||
    getType() == getMachOType(true, true);
}

void MachOObjectFile::ReadULEB128s(uint64_t Index,
                                   SmallVectorImpl<uint64_t> &Out) const {
  DataExtractor extractor(ObjectFile::getData(), true, 0);

  uint32_t offset = Index;
  uint64_t data = 0;
  while (uint64_t delta = extractor.getULEB128(&offset)) {
    data += delta;
    Out.push_back(data);
  }
}

bool MachOObjectFile::isRelocatableObject() const {
  return getHeader().filetype == MachO::MH_OBJECT;
}

ErrorOr<std::unique_ptr<MachOObjectFile>>
ObjectFile::createMachOObjectFile(MemoryBufferRef Buffer) {
  StringRef Magic = Buffer.getBuffer().slice(0, 4);
  std::error_code EC;
  std::unique_ptr<MachOObjectFile> Ret;
  if (Magic == "\xFE\xED\xFA\xCE")
    Ret.reset(new MachOObjectFile(Buffer, false, false, EC));
  else if (Magic == "\xCE\xFA\xED\xFE")
    Ret.reset(new MachOObjectFile(Buffer, true, false, EC));
  else if (Magic == "\xFE\xED\xFA\xCF")
    Ret.reset(new MachOObjectFile(Buffer, false, true, EC));
  else if (Magic == "\xCF\xFA\xED\xFE")
    Ret.reset(new MachOObjectFile(Buffer, true, true, EC));
  else
    return object_error::parse_failed;

  if (EC)
    return EC;
  return std::move(Ret);
}

