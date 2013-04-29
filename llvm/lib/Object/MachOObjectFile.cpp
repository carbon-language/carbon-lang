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
#include "llvm/ADT/Triple.h"
#include "llvm/Object/MachOFormat.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cctype>
#include <cstring>
#include <limits>

using namespace llvm;
using namespace object;

namespace llvm {
namespace object {

struct SymbolTableEntryBase {
  uint32_t StringIndex;
  uint8_t Type;
  uint8_t SectionIndex;
  uint16_t Flags;
};

struct SectionBase {
  char Name[16];
  char SegmentName[16];
};

template<typename T>
static void SwapValue(T &Value) {
  Value = sys::SwapByteOrder(Value);
}

template<typename T>
static void SwapStruct(T &Value);

template<>
void SwapStruct(macho::RelocationEntry &H) {
  SwapValue(H.Word0);
  SwapValue(H.Word1);
}

template<>
void SwapStruct(macho::LoadCommand &L) {
  SwapValue(L.Type);
  SwapValue(L.Size);
}

template<>
void SwapStruct(SymbolTableEntryBase &S) {
  SwapValue(S.StringIndex);
  SwapValue(S.Flags);
}

template<>
void SwapStruct(macho::Section &S) {
  SwapValue(S.Address);
  SwapValue(S.Size);
  SwapValue(S.Offset);
  SwapValue(S.Align);
  SwapValue(S.RelocationTableOffset);
  SwapValue(S.NumRelocationTableEntries);
  SwapValue(S.Flags);
  SwapValue(S.Reserved1);
  SwapValue(S.Reserved2);
}

template<>
void SwapStruct(macho::Section64 &S) {
  SwapValue(S.Address);
  SwapValue(S.Size);
  SwapValue(S.Offset);
  SwapValue(S.Align);
  SwapValue(S.RelocationTableOffset);
  SwapValue(S.NumRelocationTableEntries);
  SwapValue(S.Flags);
  SwapValue(S.Reserved1);
  SwapValue(S.Reserved2);
  SwapValue(S.Reserved3);
}

template<>
void SwapStruct(macho::SymbolTableEntry &S) {
  SwapValue(S.StringIndex);
  SwapValue(S.Flags);
  SwapValue(S.Value);
}

template<>
void SwapStruct(macho::Symbol64TableEntry &S) {
  SwapValue(S.StringIndex);
  SwapValue(S.Flags);
  SwapValue(S.Value);
}

template<>
void SwapStruct(macho::Header &H) {
  SwapValue(H.Magic);
  SwapValue(H.CPUType);
  SwapValue(H.CPUSubtype);
  SwapValue(H.FileType);
  SwapValue(H.NumLoadCommands);
  SwapValue(H.SizeOfLoadCommands);
  SwapValue(H.Flags);
}

template<>
void SwapStruct(macho::Header64Ext &E) {
  SwapValue(E.Reserved);
}

template<>
void SwapStruct(macho::SymtabLoadCommand &C) {
  SwapValue(C.Type);
  SwapValue(C.Size);
  SwapValue(C.SymbolTableOffset);
  SwapValue(C.NumSymbolTableEntries);
  SwapValue(C.StringTableOffset);
  SwapValue(C.StringTableSize);
}

template<>
void SwapStruct(macho::DysymtabLoadCommand &C) {
  SwapValue(C.Type);
  SwapValue(C.Size);
  SwapValue(C.LocalSymbolsIndex);
  SwapValue(C.NumLocalSymbols);
  SwapValue(C.ExternalSymbolsIndex);
  SwapValue(C.NumExternalSymbols);
  SwapValue(C.UndefinedSymbolsIndex);
  SwapValue(C.NumUndefinedSymbols);
  SwapValue(C.TOCOffset);
  SwapValue(C.NumTOCEntries);
  SwapValue(C.ModuleTableOffset);
  SwapValue(C.NumModuleTableEntries);
  SwapValue(C.ReferenceSymbolTableOffset);
  SwapValue(C.NumReferencedSymbolTableEntries);
  SwapValue(C.IndirectSymbolTableOffset);
  SwapValue(C.NumIndirectSymbolTableEntries);
  SwapValue(C.ExternalRelocationTableOffset);
  SwapValue(C.NumExternalRelocationTableEntries);
  SwapValue(C.LocalRelocationTableOffset);
  SwapValue(C.NumLocalRelocationTableEntries);
}

template<>
void SwapStruct(macho::LinkeditDataLoadCommand &C) {
  SwapValue(C.Type);
  SwapValue(C.Size);
  SwapValue(C.DataOffset);
  SwapValue(C.DataSize);
}

template<>
void SwapStruct(macho::SegmentLoadCommand &C) {
  SwapValue(C.Type);
  SwapValue(C.Size);
  SwapValue(C.VMAddress);
  SwapValue(C.VMSize);
  SwapValue(C.FileOffset);
  SwapValue(C.FileSize);
  SwapValue(C.MaxVMProtection);
  SwapValue(C.InitialVMProtection);
  SwapValue(C.NumSections);
  SwapValue(C.Flags);
}

template<>
void SwapStruct(macho::Segment64LoadCommand &C) {
  SwapValue(C.Type);
  SwapValue(C.Size);
  SwapValue(C.VMAddress);
  SwapValue(C.VMSize);
  SwapValue(C.FileOffset);
  SwapValue(C.FileSize);
  SwapValue(C.MaxVMProtection);
  SwapValue(C.InitialVMProtection);
  SwapValue(C.NumSections);
  SwapValue(C.Flags);
}

template<>
void SwapStruct(macho::IndirectSymbolTableEntry &C) {
  SwapValue(C.Index);
}

template<>
void SwapStruct(macho::LinkerOptionsLoadCommand &C) {
  SwapValue(C.Type);
  SwapValue(C.Size);
  SwapValue(C.Count);
}

template<>
void SwapStruct(macho::DataInCodeTableEntry &C) {
  SwapValue(C.Offset);
  SwapValue(C.Length);
  SwapValue(C.Kind);
}

template<typename T>
T getStruct(const MachOObjectFile *O, const char *P) {
  T Cmd;
  memcpy(&Cmd, P, sizeof(T));
  if (O->isLittleEndian() != sys::IsLittleEndianHost)
    SwapStruct(Cmd);
  return Cmd;
}

static uint32_t
getSegmentLoadCommandNumSections(const MachOObjectFile *O,
                                 const MachOObjectFile::LoadCommandInfo &L) {
  if (O->is64Bit()) {
    macho::Segment64LoadCommand S = O->getSegment64LoadCommand(L);
    return S.NumSections;
  }
  macho::SegmentLoadCommand S = O->getSegmentLoadCommand(L);
  return S.NumSections;
}

static const char *
getSectionPtr(const MachOObjectFile *O, MachOObjectFile::LoadCommandInfo L,
              unsigned Sec) {
  uintptr_t CommandAddr = reinterpret_cast<uintptr_t>(L.Ptr);

  bool Is64 = O->is64Bit();
  unsigned SegmentLoadSize = Is64 ? sizeof(macho::Segment64LoadCommand) :
                                    sizeof(macho::SegmentLoadCommand);
  unsigned SectionSize = Is64 ? sizeof(macho::Section64) :
                                sizeof(macho::Section);

  uintptr_t SectionAddr = CommandAddr + SegmentLoadSize + Sec * SectionSize;
  return reinterpret_cast<const char*>(SectionAddr);
}

static const char *getPtr(const MachOObjectFile *O, size_t Offset) {
  return O->getData().substr(Offset, 1).data();
}

static SymbolTableEntryBase
getSymbolTableEntryBase(const MachOObjectFile *O, DataRefImpl DRI) {
  const char *P = reinterpret_cast<const char *>(DRI.p);
  return getStruct<SymbolTableEntryBase>(O, P);
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
static error_code advance(T &it, size_t Val) {
  error_code ec;
  while (Val--) {
    it.increment(ec);
  }
  return ec;
}

template<class T>
static void advanceTo(T &it, size_t Val) {
  if (error_code ec = advance(it, Val))
    report_fatal_error(ec.message());
}

static unsigned getCPUType(const MachOObjectFile *O) {
  return O->getHeader().CPUType;
}

static void printRelocationTargetName(const MachOObjectFile *O,
                                      const macho::RelocationEntry &RE,
                                      raw_string_ostream &fmt) {
  bool IsScattered = O->isRelocationScattered(RE);

  // Target of a scattered relocation is an address.  In the interest of
  // generating pretty output, scan through the symbol table looking for a
  // symbol that aligns with that address.  If we find one, print it.
  // Otherwise, we just print the hex address of the target.
  if (IsScattered) {
    uint32_t Val = O->getPlainRelocationSymbolNum(RE);

    error_code ec;
    for (symbol_iterator SI = O->begin_symbols(), SE = O->end_symbols();
         SI != SE; SI.increment(ec)) {
      if (ec) report_fatal_error(ec.message());

      uint64_t Addr;
      StringRef Name;

      if ((ec = SI->getAddress(Addr)))
        report_fatal_error(ec.message());
      if (Addr != Val) continue;
      if ((ec = SI->getName(Name)))
        report_fatal_error(ec.message());
      fmt << Name;
      return;
    }

    // If we couldn't find a symbol that this relocation refers to, try
    // to find a section beginning instead.
    for (section_iterator SI = O->begin_sections(), SE = O->end_sections();
         SI != SE; SI.increment(ec)) {
      if (ec) report_fatal_error(ec.message());

      uint64_t Addr;
      StringRef Name;

      if ((ec = SI->getAddress(Addr)))
        report_fatal_error(ec.message());
      if (Addr != Val) continue;
      if ((ec = SI->getName(Name)))
        report_fatal_error(ec.message());
      fmt << Name;
      return;
    }

    fmt << format("0x%x", Val);
    return;
  }

  StringRef S;
  bool isExtern = O->getPlainRelocationExternal(RE);
  uint64_t Val = O->getAnyRelocationAddress(RE);

  if (isExtern) {
    symbol_iterator SI = O->begin_symbols();
    advanceTo(SI, Val);
    SI->getName(S);
  } else {
    section_iterator SI = O->begin_sections();
    advanceTo(SI, Val);
    SI->getName(S);
  }

  fmt << S;
}

static uint32_t getPlainRelocationAddress(const macho::RelocationEntry &RE) {
  return RE.Word0;
}

static unsigned
getScatteredRelocationAddress(const macho::RelocationEntry &RE) {
  return RE.Word0 & 0xffffff;
}

static bool getPlainRelocationPCRel(const MachOObjectFile *O,
                                    const macho::RelocationEntry &RE) {
  if (O->isLittleEndian())
    return (RE.Word1 >> 24) & 1;
  return (RE.Word1 >> 7) & 1;
}

static bool
getScatteredRelocationPCRel(const MachOObjectFile *O,
                            const macho::RelocationEntry &RE) {
  return (RE.Word0 >> 30) & 1;
}

static unsigned getPlainRelocationLength(const MachOObjectFile *O,
                                         const macho::RelocationEntry &RE) {
  if (O->isLittleEndian())
    return (RE.Word1 >> 25) & 3;
  return (RE.Word1 >> 5) & 3;
}

static unsigned
getScatteredRelocationLength(const macho::RelocationEntry &RE) {
  return (RE.Word0 >> 28) & 3;
}

static unsigned getPlainRelocationType(const MachOObjectFile *O,
                                       const macho::RelocationEntry &RE) {
  if (O->isLittleEndian())
    return RE.Word1 >> 28;
  return RE.Word1 & 0xf;
}

static unsigned getScatteredRelocationType(const macho::RelocationEntry &RE) {
  return (RE.Word0 >> 24) & 0xf;
}

static uint32_t getSectionFlags(const MachOObjectFile *O,
                                DataRefImpl Sec) {
  if (O->is64Bit()) {
    macho::Section64 Sect = O->getSection64(Sec);
    return Sect.Flags;
  }
  macho::Section Sect = O->getSection(Sec);
  return Sect.Flags;
}

MachOObjectFile::MachOObjectFile(MemoryBuffer *Object,
                                 bool IsLittleEndian, bool Is64bits,
                                 error_code &ec)
    : ObjectFile(getMachOType(IsLittleEndian, Is64bits), Object),
      SymtabLoadCmd(NULL), DysymtabLoadCmd(NULL) {
  uint32_t LoadCommandCount = this->getHeader().NumLoadCommands;
  macho::LoadCommandType SegmentLoadType = is64Bit() ?
    macho::LCT_Segment64 : macho::LCT_Segment;

  MachOObjectFile::LoadCommandInfo Load = getFirstLoadCommandInfo();
  for (unsigned I = 0; ; ++I) {
    if (Load.C.Type == macho::LCT_Symtab) {
      assert(!SymtabLoadCmd && "Multiple symbol tables");
      SymtabLoadCmd = Load.Ptr;
    } else if (Load.C.Type == macho::LCT_Dysymtab) {
      assert(!DysymtabLoadCmd && "Multiple dynamic symbol tables");
      DysymtabLoadCmd = Load.Ptr;
    } else if (Load.C.Type == SegmentLoadType) {
      uint32_t NumSections = getSegmentLoadCommandNumSections(this, Load);
      for (unsigned J = 0; J < NumSections; ++J) {
        const char *Sec = getSectionPtr(this, Load, J);
        Sections.push_back(Sec);
      }
    }

    if (I == LoadCommandCount - 1)
      break;
    else
      Load = getNextLoadCommandInfo(Load);
  }
}

error_code MachOObjectFile::getSymbolNext(DataRefImpl Symb,
                                          SymbolRef &Res) const {
  unsigned SymbolTableEntrySize = is64Bit() ?
    sizeof(macho::Symbol64TableEntry) :
    sizeof(macho::SymbolTableEntry);
  Symb.p += SymbolTableEntrySize;
  Res = SymbolRef(Symb, this);
  return object_error::success;
}

error_code MachOObjectFile::getSymbolName(DataRefImpl Symb,
                                          StringRef &Res) const {
  StringRef StringTable = getStringTableData();
  SymbolTableEntryBase Entry = getSymbolTableEntryBase(this, Symb);
  const char *Start = &StringTable.data()[Entry.StringIndex];
  Res = StringRef(Start);
  return object_error::success;
}

error_code MachOObjectFile::getSymbolAddress(DataRefImpl Symb,
                                             uint64_t &Res) const {
  if (is64Bit()) {
    macho::Symbol64TableEntry Entry = getSymbol64TableEntry(Symb);
    Res = Entry.Value;
  } else {
    macho::SymbolTableEntry Entry = getSymbolTableEntry(Symb);
    Res = Entry.Value;
  }
  return object_error::success;
}

error_code
MachOObjectFile::getSymbolFileOffset(DataRefImpl Symb,
                                     uint64_t &Res) const {
  SymbolTableEntryBase Entry = getSymbolTableEntryBase(this, Symb);
  getSymbolAddress(Symb, Res);
  if (Entry.SectionIndex) {
    uint64_t Delta;
    DataRefImpl SecRel;
    SecRel.d.a = Entry.SectionIndex-1;
    if (is64Bit()) {
      macho::Section64 Sec = getSection64(SecRel);
      Delta = Sec.Offset - Sec.Address;
    } else {
      macho::Section Sec = getSection(SecRel);
      Delta = Sec.Offset - Sec.Address;
    }

    Res += Delta;
  }

  return object_error::success;
}

error_code MachOObjectFile::getSymbolAlignment(DataRefImpl DRI,
                                               uint32_t &Result) const {
  uint32_t flags;
  this->getSymbolFlags(DRI, flags);
  if (flags & SymbolRef::SF_Common) {
    SymbolTableEntryBase Entry = getSymbolTableEntryBase(this, DRI);
    Result = 1 << MachO::GET_COMM_ALIGN(Entry.Flags);
  } else {
    Result = 0;
  }
  return object_error::success;
}

error_code MachOObjectFile::getSymbolSize(DataRefImpl DRI,
                                          uint64_t &Result) const {
  uint64_t BeginOffset;
  uint64_t EndOffset = 0;
  uint8_t SectionIndex;

  SymbolTableEntryBase Entry = getSymbolTableEntryBase(this, DRI);
  uint64_t Value;
  getSymbolAddress(DRI, Value);

  BeginOffset = Value;

  SectionIndex = Entry.SectionIndex;
  if (!SectionIndex) {
    uint32_t flags = SymbolRef::SF_None;
    this->getSymbolFlags(DRI, flags);
    if (flags & SymbolRef::SF_Common)
      Result = Value;
    else
      Result = UnknownAddressOrSize;
    return object_error::success;
  }
  // Unfortunately symbols are unsorted so we need to touch all
  // symbols from load command
  error_code ec;
  for (symbol_iterator I = begin_symbols(), E = end_symbols(); I != E;
       I.increment(ec)) {
    DataRefImpl DRI = I->getRawDataRefImpl();
    Entry = getSymbolTableEntryBase(this, DRI);
    getSymbolAddress(DRI, Value);
    if (Entry.SectionIndex == SectionIndex && Value > BeginOffset)
      if (!EndOffset || Value < EndOffset)
        EndOffset = Value;
  }
  if (!EndOffset) {
    uint64_t Size;
    DataRefImpl Sec;
    Sec.d.a = SectionIndex-1;
    getSectionSize(Sec, Size);
    getSectionAddress(Sec, EndOffset);
    EndOffset += Size;
  }
  Result = EndOffset - BeginOffset;
  return object_error::success;
}

error_code MachOObjectFile::getSymbolType(DataRefImpl Symb,
                                          SymbolRef::Type &Res) const {
  SymbolTableEntryBase Entry = getSymbolTableEntryBase(this, Symb);
  uint8_t n_type = Entry.Type;

  Res = SymbolRef::ST_Other;

  // If this is a STAB debugging symbol, we can do nothing more.
  if (n_type & MachO::NlistMaskStab) {
    Res = SymbolRef::ST_Debug;
    return object_error::success;
  }

  switch (n_type & MachO::NlistMaskType) {
    case MachO::NListTypeUndefined :
      Res = SymbolRef::ST_Unknown;
      break;
    case MachO::NListTypeSection :
      Res = SymbolRef::ST_Function;
      break;
  }
  return object_error::success;
}

error_code MachOObjectFile::getSymbolNMTypeChar(DataRefImpl Symb,
                                                char &Res) const {
  SymbolTableEntryBase Entry = getSymbolTableEntryBase(this, Symb);
  uint8_t Type = Entry.Type;
  uint16_t Flags = Entry.Flags;

  char Char;
  switch (Type & macho::STF_TypeMask) {
    case macho::STT_Undefined:
      Char = 'u';
      break;
    case macho::STT_Absolute:
    case macho::STT_Section:
      Char = 's';
      break;
    default:
      Char = '?';
      break;
  }

  if (Flags & (macho::STF_External | macho::STF_PrivateExtern))
    Char = toupper(static_cast<unsigned char>(Char));
  Res = Char;
  return object_error::success;
}

error_code MachOObjectFile::getSymbolFlags(DataRefImpl DRI,
                                           uint32_t &Result) const {
  SymbolTableEntryBase Entry = getSymbolTableEntryBase(this, DRI);

  uint8_t MachOType = Entry.Type;
  uint16_t MachOFlags = Entry.Flags;

  // TODO: Correctly set SF_ThreadLocal
  Result = SymbolRef::SF_None;

  if ((MachOType & MachO::NlistMaskType) == MachO::NListTypeUndefined)
    Result |= SymbolRef::SF_Undefined;

  if (MachOFlags & macho::STF_StabsEntryMask)
    Result |= SymbolRef::SF_FormatSpecific;

  if (MachOType & MachO::NlistMaskExternal) {
    Result |= SymbolRef::SF_Global;
    if ((MachOType & MachO::NlistMaskType) == MachO::NListTypeUndefined) {
      uint64_t Value;
      getSymbolAddress(DRI, Value);
      if (Value)
        Result |= SymbolRef::SF_Common;
    }
  }

  if (MachOFlags & (MachO::NListDescWeakRef | MachO::NListDescWeakDef))
    Result |= SymbolRef::SF_Weak;

  if ((MachOType & MachO::NlistMaskType) == MachO::NListTypeAbsolute)
    Result |= SymbolRef::SF_Absolute;

  return object_error::success;
}

error_code
MachOObjectFile::getSymbolSection(DataRefImpl Symb,
                                  section_iterator &Res) const {
  SymbolTableEntryBase Entry = getSymbolTableEntryBase(this, Symb);
  uint8_t index = Entry.SectionIndex;

  if (index == 0) {
    Res = end_sections();
  } else {
    DataRefImpl DRI;
    DRI.d.a = index - 1;
    Res = section_iterator(SectionRef(DRI, this));
  }

  return object_error::success;
}

error_code MachOObjectFile::getSymbolValue(DataRefImpl Symb,
                                           uint64_t &Val) const {
  report_fatal_error("getSymbolValue unimplemented in MachOObjectFile");
}

error_code MachOObjectFile::getSectionNext(DataRefImpl Sec,
                                           SectionRef &Res) const {
  Sec.d.a++;
  Res = SectionRef(Sec, this);
  return object_error::success;
}

error_code
MachOObjectFile::getSectionName(DataRefImpl Sec, StringRef &Result) const {
  ArrayRef<char> Raw = getSectionRawName(Sec);
  Result = parseSegmentOrSectionName(Raw.data());
  return object_error::success;
}

error_code
MachOObjectFile::getSectionAddress(DataRefImpl Sec, uint64_t &Res) const {
  if (is64Bit()) {
    macho::Section64 Sect = getSection64(Sec);
    Res = Sect.Address;
  } else {
    macho::Section Sect = getSection(Sec);
    Res = Sect.Address;
  }
  return object_error::success;
}

error_code
MachOObjectFile::getSectionSize(DataRefImpl Sec, uint64_t &Res) const {
  if (is64Bit()) {
    macho::Section64 Sect = getSection64(Sec);
    Res = Sect.Size;
  } else {
    macho::Section Sect = getSection(Sec);
    Res = Sect.Size;
  }

  return object_error::success;
}

error_code
MachOObjectFile::getSectionContents(DataRefImpl Sec, StringRef &Res) const {
  uint32_t Offset;
  uint64_t Size;

  if (is64Bit()) {
    macho::Section64 Sect = getSection64(Sec);
    Offset = Sect.Offset;
    Size = Sect.Size;
  } else {
    macho::Section Sect =getSection(Sec);
    Offset = Sect.Offset;
    Size = Sect.Size;
  }

  Res = this->getData().substr(Offset, Size);
  return object_error::success;
}

error_code
MachOObjectFile::getSectionAlignment(DataRefImpl Sec, uint64_t &Res) const {
  uint32_t Align;
  if (is64Bit()) {
    macho::Section64 Sect = getSection64(Sec);
    Align = Sect.Align;
  } else {
    macho::Section Sect = getSection(Sec);
    Align = Sect.Align;
  }

  Res = uint64_t(1) << Align;
  return object_error::success;
}

error_code
MachOObjectFile::isSectionText(DataRefImpl Sec, bool &Res) const {
  uint32_t Flags = getSectionFlags(this, Sec);
  Res = Flags & macho::SF_PureInstructions;
  return object_error::success;
}

error_code MachOObjectFile::isSectionData(DataRefImpl DRI, bool &Result) const {
  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

error_code MachOObjectFile::isSectionBSS(DataRefImpl DRI, bool &Result) const {
  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

error_code
MachOObjectFile::isSectionRequiredForExecution(DataRefImpl Sec,
                                               bool &Result) const {
  // FIXME: Unimplemented.
  Result = true;
  return object_error::success;
}

error_code MachOObjectFile::isSectionVirtual(DataRefImpl Sec,
                                             bool &Result) const {
  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

error_code
MachOObjectFile::isSectionZeroInit(DataRefImpl Sec, bool &Res) const {
  uint32_t Flags = getSectionFlags(this, Sec);
  unsigned SectionType = Flags & MachO::SectionFlagMaskSectionType;
  Res = SectionType == MachO::SectionTypeZeroFill ||
    SectionType == MachO::SectionTypeZeroFillLarge;
  return object_error::success;
}

error_code MachOObjectFile::isSectionReadOnlyData(DataRefImpl Sec,
                                                  bool &Result) const {
  // Consider using the code from isSectionText to look for __const sections.
  // Alternately, emit S_ATTR_PURE_INSTRUCTIONS and/or S_ATTR_SOME_INSTRUCTIONS
  // to use section attributes to distinguish code from data.

  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

error_code
MachOObjectFile::sectionContainsSymbol(DataRefImpl Sec, DataRefImpl Symb,
                                       bool &Result) const {
  SymbolRef::Type ST;
  this->getSymbolType(Symb, ST);
  if (ST == SymbolRef::ST_Unknown) {
    Result = false;
    return object_error::success;
  }

  uint64_t SectBegin, SectEnd;
  getSectionAddress(Sec, SectBegin);
  getSectionSize(Sec, SectEnd);
  SectEnd += SectBegin;

  uint64_t SymAddr;
  getSymbolAddress(Symb, SymAddr);
  Result = (SymAddr >= SectBegin) && (SymAddr < SectEnd);

  return object_error::success;
}

relocation_iterator MachOObjectFile::getSectionRelBegin(DataRefImpl Sec) const {
  uint32_t Offset;
  if (is64Bit()) {
    macho::Section64 Sect = getSection64(Sec);
    Offset = Sect.RelocationTableOffset;
  } else {
    macho::Section Sect = getSection(Sec);
    Offset = Sect.RelocationTableOffset;
  }

  DataRefImpl Ret;
  Ret.p = reinterpret_cast<uintptr_t>(getPtr(this, Offset));
  return relocation_iterator(RelocationRef(Ret, this));
}

relocation_iterator
MachOObjectFile::getSectionRelEnd(DataRefImpl Sec) const {
  uint32_t Offset;
  uint32_t Num;
  if (is64Bit()) {
    macho::Section64 Sect = getSection64(Sec);
    Offset = Sect.RelocationTableOffset;
    Num = Sect.NumRelocationTableEntries;
  } else {
    macho::Section Sect = getSection(Sec);
    Offset = Sect.RelocationTableOffset;
    Num = Sect.NumRelocationTableEntries;
  }

  const macho::RelocationEntry *P =
    reinterpret_cast<const macho::RelocationEntry*>(getPtr(this, Offset));

  DataRefImpl Ret;
  Ret.p = reinterpret_cast<uintptr_t>(P + Num);
  return relocation_iterator(RelocationRef(Ret, this));
}

error_code MachOObjectFile::getRelocationNext(DataRefImpl Rel,
                                              RelocationRef &Res) const {
  const macho::RelocationEntry *P =
    reinterpret_cast<const macho::RelocationEntry *>(Rel.p);
  Rel.p = reinterpret_cast<uintptr_t>(P + 1);
  Res = RelocationRef(Rel, this);
  return object_error::success;
}

error_code
MachOObjectFile::getRelocationAddress(DataRefImpl Rel, uint64_t &Res) const {
  report_fatal_error("getRelocationAddress not implemented in MachOObjectFile");
}

error_code MachOObjectFile::getRelocationOffset(DataRefImpl Rel,
                                                uint64_t &Res) const {
  macho::RelocationEntry RE = getRelocation(Rel);
  Res = getAnyRelocationAddress(RE);
  return object_error::success;
}

error_code
MachOObjectFile::getRelocationSymbol(DataRefImpl Rel, SymbolRef &Res) const {
  macho::RelocationEntry RE = getRelocation(Rel);
  uint32_t SymbolIdx = getPlainRelocationSymbolNum(RE);
  bool isExtern = getPlainRelocationExternal(RE);
  if (!isExtern) {
    Res = *end_symbols();
    return object_error::success;
  }

  macho::SymtabLoadCommand S = getSymtabLoadCommand();
  unsigned SymbolTableEntrySize = is64Bit() ?
    sizeof(macho::Symbol64TableEntry) :
    sizeof(macho::SymbolTableEntry);
  uint64_t Offset = S.SymbolTableOffset + SymbolIdx * SymbolTableEntrySize;
  DataRefImpl Sym;
  Sym.p = reinterpret_cast<uintptr_t>(getPtr(this, Offset));
  Res = SymbolRef(Sym, this);
  return object_error::success;
}

error_code MachOObjectFile::getRelocationType(DataRefImpl Rel,
                                              uint64_t &Res) const {
  macho::RelocationEntry RE = getRelocation(Rel);
  Res = getAnyRelocationType(RE);
  return object_error::success;
}

error_code
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

      if (RType > 6)
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

error_code MachOObjectFile::getRelocationAdditionalInfo(DataRefImpl Rel,
                                                        int64_t &Res) const {
  Res = 0;
  return object_error::success;
}

error_code
MachOObjectFile::getRelocationValueString(DataRefImpl Rel,
                                          SmallVectorImpl<char> &Result) const {
  macho::RelocationEntry RE = getRelocation(Rel);

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
      case macho::RIT_X86_64_GOTLoad:   // X86_64_RELOC_GOT_LOAD
      case macho::RIT_X86_64_GOT: {     // X86_64_RELOC_GOT
        printRelocationTargetName(this, RE, fmt);
        fmt << "@GOT";
        if (isPCRel) fmt << "PCREL";
        break;
      }
      case macho::RIT_X86_64_Subtractor: { // X86_64_RELOC_SUBTRACTOR
        DataRefImpl RelNext = Rel;
        RelNext.d.a++;
        macho::RelocationEntry RENext = getRelocation(RelNext);

        // X86_64_SUBTRACTOR must be followed by a relocation of type
        // X86_64_RELOC_UNSIGNED.
        // NOTE: Scattered relocations don't exist on x86_64.
        unsigned RType = getAnyRelocationType(RENext);
        if (RType != 0)
          report_fatal_error("Expected X86_64_RELOC_UNSIGNED after "
                             "X86_64_RELOC_SUBTRACTOR.");

        // The X86_64_RELOC_UNSIGNED contains the minuend symbol,
        // X86_64_SUBTRACTOR contains to the subtrahend.
        printRelocationTargetName(this, RENext, fmt);
        fmt << "-";
        printRelocationTargetName(this, RE, fmt);
        break;
      }
      case macho::RIT_X86_64_TLV:
        printRelocationTargetName(this, RE, fmt);
        fmt << "@TLV";
        if (isPCRel) fmt << "P";
        break;
      case macho::RIT_X86_64_Signed1: // X86_64_RELOC_SIGNED1
        printRelocationTargetName(this, RE, fmt);
        fmt << "-1";
        break;
      case macho::RIT_X86_64_Signed2: // X86_64_RELOC_SIGNED2
        printRelocationTargetName(this, RE, fmt);
        fmt << "-2";
        break;
      case macho::RIT_X86_64_Signed4: // X86_64_RELOC_SIGNED4
        printRelocationTargetName(this, RE, fmt);
        fmt << "-4";
        break;
      default:
        printRelocationTargetName(this, RE, fmt);
        break;
    }
  // X86 and ARM share some relocation types in common.
  } else if (Arch == Triple::x86 || Arch == Triple::arm) {
    // Generic relocation types...
    switch (Type) {
      case macho::RIT_Pair: // GENERIC_RELOC_PAIR - prints no info
        return object_error::success;
      case macho::RIT_Difference: { // GENERIC_RELOC_SECTDIFF
        DataRefImpl RelNext = Rel;
        RelNext.d.a++;
        macho::RelocationEntry RENext = getRelocation(RelNext);

        // X86 sect diff's must be followed by a relocation of type
        // GENERIC_RELOC_PAIR.
        unsigned RType = getAnyRelocationType(RENext);

        if (RType != 1)
          report_fatal_error("Expected GENERIC_RELOC_PAIR after "
                             "GENERIC_RELOC_SECTDIFF.");

        printRelocationTargetName(this, RE, fmt);
        fmt << "-";
        printRelocationTargetName(this, RENext, fmt);
        break;
      }
    }

    if (Arch == Triple::x86) {
      // All X86 relocations that need special printing were already
      // handled in the generic code.
      switch (Type) {
        case macho::RIT_Generic_LocalDifference:{// GENERIC_RELOC_LOCAL_SECTDIFF
          DataRefImpl RelNext = Rel;
          RelNext.d.a++;
          macho::RelocationEntry RENext = getRelocation(RelNext);

          // X86 sect diff's must be followed by a relocation of type
          // GENERIC_RELOC_PAIR.
          unsigned RType = getAnyRelocationType(RENext);
          if (RType != 1)
            report_fatal_error("Expected GENERIC_RELOC_PAIR after "
                               "GENERIC_RELOC_LOCAL_SECTDIFF.");

          printRelocationTargetName(this, RE, fmt);
          fmt << "-";
          printRelocationTargetName(this, RENext, fmt);
          break;
        }
        case macho::RIT_Generic_TLV: {
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
        case macho::RIT_ARM_Half:             // ARM_RELOC_HALF
        case macho::RIT_ARM_HalfDifference: { // ARM_RELOC_HALF_SECTDIFF
          // Half relocations steal a bit from the length field to encode
          // whether this is an upper16 or a lower16 relocation.
          bool isUpper = getAnyRelocationLength(RE) >> 1;

          if (isUpper)
            fmt << ":upper16:(";
          else
            fmt << ":lower16:(";
          printRelocationTargetName(this, RE, fmt);

          DataRefImpl RelNext = Rel;
          RelNext.d.a++;
          macho::RelocationEntry RENext = getRelocation(RelNext);

          // ARM half relocs must be followed by a relocation of type
          // ARM_RELOC_PAIR.
          unsigned RType = getAnyRelocationType(RENext);
          if (RType != 1)
            report_fatal_error("Expected ARM_RELOC_PAIR after "
                               "GENERIC_RELOC_HALF");

          // NOTE: The half of the target virtual address is stashed in the
          // address field of the secondary relocation, but we can't reverse
          // engineer the constant offset from it without decoding the movw/movt
          // instruction to find the other half in its immediate field.

          // ARM_RELOC_HALF_SECTDIFF encodes the second section in the
          // symbol/section pointer of the follow-on relocation.
          if (Type == macho::RIT_ARM_HalfDifference) {
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

error_code
MachOObjectFile::getRelocationHidden(DataRefImpl Rel, bool &Result) const {
  unsigned Arch = getArch();
  uint64_t Type;
  getRelocationType(Rel, Type);

  Result = false;

  // On arches that use the generic relocations, GENERIC_RELOC_PAIR
  // is always hidden.
  if (Arch == Triple::x86 || Arch == Triple::arm) {
    if (Type == macho::RIT_Pair) Result = true;
  } else if (Arch == Triple::x86_64) {
    // On x86_64, X86_64_RELOC_UNSIGNED is hidden only when it follows
    // an X864_64_RELOC_SUBTRACTOR.
    if (Type == macho::RIT_X86_64_Unsigned && Rel.d.a > 0) {
      DataRefImpl RelPrev = Rel;
      RelPrev.d.a--;
      uint64_t PrevType;
      getRelocationType(RelPrev, PrevType);
      if (PrevType == macho::RIT_X86_64_Subtractor)
        Result = true;
    }
  }

  return object_error::success;
}

error_code MachOObjectFile::getLibraryNext(DataRefImpl LibData,
                                           LibraryRef &Res) const {
  report_fatal_error("Needed libraries unimplemented in MachOObjectFile");
}

error_code MachOObjectFile::getLibraryPath(DataRefImpl LibData,
                                           StringRef &Res) const {
  report_fatal_error("Needed libraries unimplemented in MachOObjectFile");
}

symbol_iterator MachOObjectFile::begin_symbols() const {
  DataRefImpl DRI;
  if (!SymtabLoadCmd)
    return symbol_iterator(SymbolRef(DRI, this));

  macho::SymtabLoadCommand Symtab = getSymtabLoadCommand();
  DRI.p = reinterpret_cast<uintptr_t>(getPtr(this, Symtab.SymbolTableOffset));
  return symbol_iterator(SymbolRef(DRI, this));
}

symbol_iterator MachOObjectFile::end_symbols() const {
  DataRefImpl DRI;
  if (!SymtabLoadCmd)
    return symbol_iterator(SymbolRef(DRI, this));

  macho::SymtabLoadCommand Symtab = getSymtabLoadCommand();
  unsigned SymbolTableEntrySize = is64Bit() ?
    sizeof(macho::Symbol64TableEntry) :
    sizeof(macho::SymbolTableEntry);
  unsigned Offset = Symtab.SymbolTableOffset +
    Symtab.NumSymbolTableEntries * SymbolTableEntrySize;
  DRI.p = reinterpret_cast<uintptr_t>(getPtr(this, Offset));
  return symbol_iterator(SymbolRef(DRI, this));
}

symbol_iterator MachOObjectFile::begin_dynamic_symbols() const {
  // TODO: implement
  report_fatal_error("Dynamic symbols unimplemented in MachOObjectFile");
}

symbol_iterator MachOObjectFile::end_dynamic_symbols() const {
  // TODO: implement
  report_fatal_error("Dynamic symbols unimplemented in MachOObjectFile");
}

section_iterator MachOObjectFile::begin_sections() const {
  DataRefImpl DRI;
  return section_iterator(SectionRef(DRI, this));
}

section_iterator MachOObjectFile::end_sections() const {
  DataRefImpl DRI;
  DRI.d.a = Sections.size();
  return section_iterator(SectionRef(DRI, this));
}

library_iterator MachOObjectFile::begin_libraries_needed() const {
  // TODO: implement
  report_fatal_error("Needed libraries unimplemented in MachOObjectFile");
}

library_iterator MachOObjectFile::end_libraries_needed() const {
  // TODO: implement
  report_fatal_error("Needed libraries unimplemented in MachOObjectFile");
}

uint8_t MachOObjectFile::getBytesInAddress() const {
  return is64Bit() ? 8 : 4;
}

StringRef MachOObjectFile::getFileFormatName() const {
  unsigned CPUType = getCPUType(this);
  if (!is64Bit()) {
    switch (CPUType) {
    case llvm::MachO::CPUTypeI386:
      return "Mach-O 32-bit i386";
    case llvm::MachO::CPUTypeARM:
      return "Mach-O arm";
    case llvm::MachO::CPUTypePowerPC:
      return "Mach-O 32-bit ppc";
    default:
      assert((CPUType & llvm::MachO::CPUArchABI64) == 0 &&
             "64-bit object file when we're not 64-bit?");
      return "Mach-O 32-bit unknown";
    }
  }

  // Make sure the cpu type has the correct mask.
  assert((CPUType & llvm::MachO::CPUArchABI64)
	 == llvm::MachO::CPUArchABI64 &&
	 "32-bit object file when we're 64-bit?");

  switch (CPUType) {
  case llvm::MachO::CPUTypeX86_64:
    return "Mach-O 64-bit x86-64";
  case llvm::MachO::CPUTypePowerPC64:
    return "Mach-O 64-bit ppc64";
  default:
    return "Mach-O 64-bit unknown";
  }
}

unsigned MachOObjectFile::getArch() const {
  switch (getCPUType(this)) {
  case llvm::MachO::CPUTypeI386:
    return Triple::x86;
  case llvm::MachO::CPUTypeX86_64:
    return Triple::x86_64;
  case llvm::MachO::CPUTypeARM:
    return Triple::arm;
  case llvm::MachO::CPUTypePowerPC:
    return Triple::ppc;
  case llvm::MachO::CPUTypePowerPC64:
    return Triple::ppc64;
  default:
    return Triple::UnknownArch;
  }
}

StringRef MachOObjectFile::getLoadName() const {
  // TODO: Implement
  report_fatal_error("get_load_name() unimplemented in MachOObjectFile");
}

relocation_iterator MachOObjectFile::getSectionRelBegin(unsigned Index) const {
  DataRefImpl DRI;
  DRI.d.a = Index;
  return getSectionRelBegin(DRI);
}

relocation_iterator MachOObjectFile::getSectionRelEnd(unsigned Index) const {
  DataRefImpl DRI;
  DRI.d.a = Index;
  return getSectionRelEnd(DRI);
}

StringRef
MachOObjectFile::getSectionFinalSegmentName(DataRefImpl Sec) const {
  ArrayRef<char> Raw = getSectionRawFinalSegmentName(Sec);
  return parseSegmentOrSectionName(Raw.data());
}

ArrayRef<char>
MachOObjectFile::getSectionRawName(DataRefImpl Sec) const {
  const SectionBase *Base =
    reinterpret_cast<const SectionBase*>(Sections[Sec.d.a]);
  return ArrayRef<char>(Base->Name);
}

ArrayRef<char>
MachOObjectFile::getSectionRawFinalSegmentName(DataRefImpl Sec) const {
  const SectionBase *Base =
    reinterpret_cast<const SectionBase*>(Sections[Sec.d.a]);
  return ArrayRef<char>(Base->SegmentName);
}

bool
MachOObjectFile::isRelocationScattered(const macho::RelocationEntry &RE)
  const {
  if (getCPUType(this) == llvm::MachO::CPUTypeX86_64)
    return false;
  return getPlainRelocationAddress(RE) & macho::RF_Scattered;
}

unsigned MachOObjectFile::getPlainRelocationSymbolNum(const macho::RelocationEntry &RE) const {
  if (isLittleEndian())
    return RE.Word1 & 0xffffff;
  return RE.Word1 >> 8;
}

bool MachOObjectFile::getPlainRelocationExternal(const macho::RelocationEntry &RE) const {
  if (isLittleEndian())
    return (RE.Word1 >> 27) & 1;
  return (RE.Word1 >> 4) & 1;
}

bool
MachOObjectFile::getScatteredRelocationScattered(const macho::RelocationEntry &RE) const {
  return RE.Word0 >> 31;
}

uint32_t
MachOObjectFile::getScatteredRelocationValue(const macho::RelocationEntry &RE) const {
  return RE.Word1;
}

unsigned
MachOObjectFile::getAnyRelocationAddress(const macho::RelocationEntry &RE) const {
  if (isRelocationScattered(RE))
    return getScatteredRelocationAddress(RE);
  return getPlainRelocationAddress(RE);
}

unsigned
MachOObjectFile::getAnyRelocationPCRel(const macho::RelocationEntry &RE) const {
  if (isRelocationScattered(RE))
    return getScatteredRelocationPCRel(this, RE);
  return getPlainRelocationPCRel(this, RE);
}

unsigned
MachOObjectFile::getAnyRelocationLength(const macho::RelocationEntry &RE) const {
  if (isRelocationScattered(RE))
    return getScatteredRelocationLength(RE);
  return getPlainRelocationLength(this, RE);
}

unsigned
MachOObjectFile::getAnyRelocationType(const macho::RelocationEntry &RE) const {
  if (isRelocationScattered(RE))
    return getScatteredRelocationType(RE);
  return getPlainRelocationType(this, RE);
}

MachOObjectFile::LoadCommandInfo
MachOObjectFile::getFirstLoadCommandInfo() const {
  MachOObjectFile::LoadCommandInfo Load;

  unsigned HeaderSize = is64Bit() ? macho::Header64Size : macho::Header32Size;
  Load.Ptr = getPtr(this, HeaderSize);
  Load.C = getStruct<macho::LoadCommand>(this, Load.Ptr);
  return Load;
}

MachOObjectFile::LoadCommandInfo
MachOObjectFile::getNextLoadCommandInfo(const LoadCommandInfo &L) const {
  MachOObjectFile::LoadCommandInfo Next;
  Next.Ptr = L.Ptr + L.C.Size;
  Next.C = getStruct<macho::LoadCommand>(this, Next.Ptr);
  return Next;
}

macho::Section MachOObjectFile::getSection(DataRefImpl DRI) const {
  return getStruct<macho::Section>(this, Sections[DRI.d.a]);
}

macho::Section64 MachOObjectFile::getSection64(DataRefImpl DRI) const {
  return getStruct<macho::Section64>(this, Sections[DRI.d.a]);
}

macho::Section MachOObjectFile::getSection(const LoadCommandInfo &L,
                                           unsigned Index) const {
  const char *Sec = getSectionPtr(this, L, Index);
  return getStruct<macho::Section>(this, Sec);
}

macho::Section64 MachOObjectFile::getSection64(const LoadCommandInfo &L,
                                               unsigned Index) const {
  const char *Sec = getSectionPtr(this, L, Index);
  return getStruct<macho::Section64>(this, Sec);
}

macho::SymbolTableEntry
MachOObjectFile::getSymbolTableEntry(DataRefImpl DRI) const {
  const char *P = reinterpret_cast<const char *>(DRI.p);
  return getStruct<macho::SymbolTableEntry>(this, P);
}

macho::Symbol64TableEntry
MachOObjectFile::getSymbol64TableEntry(DataRefImpl DRI) const {
  const char *P = reinterpret_cast<const char *>(DRI.p);
  return getStruct<macho::Symbol64TableEntry>(this, P);
}

macho::LinkeditDataLoadCommand
MachOObjectFile::getLinkeditDataLoadCommand(const MachOObjectFile::LoadCommandInfo &L) const {
  return getStruct<macho::LinkeditDataLoadCommand>(this, L.Ptr);
}

macho::SegmentLoadCommand
MachOObjectFile::getSegmentLoadCommand(const LoadCommandInfo &L) const {
  return getStruct<macho::SegmentLoadCommand>(this, L.Ptr);
}

macho::Segment64LoadCommand
MachOObjectFile::getSegment64LoadCommand(const LoadCommandInfo &L) const {
  return getStruct<macho::Segment64LoadCommand>(this, L.Ptr);
}

macho::LinkerOptionsLoadCommand
MachOObjectFile::getLinkerOptionsLoadCommand(const LoadCommandInfo &L) const {
  return getStruct<macho::LinkerOptionsLoadCommand>(this, L.Ptr);
}

macho::RelocationEntry
MachOObjectFile::getRelocation(DataRefImpl Rel) const {
  const char *P = reinterpret_cast<const char *>(Rel.p);
  return getStruct<macho::RelocationEntry>(this, P);
}

macho::Header MachOObjectFile::getHeader() const {
  return getStruct<macho::Header>(this, getPtr(this, 0));
}

macho::Header64Ext MachOObjectFile::getHeader64Ext() const {
  return
    getStruct<macho::Header64Ext>(this, getPtr(this, sizeof(macho::Header)));
}

macho::IndirectSymbolTableEntry MachOObjectFile::getIndirectSymbolTableEntry(
                                          const macho::DysymtabLoadCommand &DLC,
                                          unsigned Index) const {
  uint64_t Offset = DLC.IndirectSymbolTableOffset +
    Index * sizeof(macho::IndirectSymbolTableEntry);
  return getStruct<macho::IndirectSymbolTableEntry>(this, getPtr(this, Offset));
}

macho::DataInCodeTableEntry
MachOObjectFile::getDataInCodeTableEntry(uint32_t DataOffset,
                                         unsigned Index) const {
  uint64_t Offset = DataOffset + Index * sizeof(macho::DataInCodeTableEntry);
  return getStruct<macho::DataInCodeTableEntry>(this, getPtr(this, Offset));
}

macho::SymtabLoadCommand MachOObjectFile::getSymtabLoadCommand() const {
  return getStruct<macho::SymtabLoadCommand>(this, SymtabLoadCmd);
}

macho::DysymtabLoadCommand MachOObjectFile::getDysymtabLoadCommand() const {
  return getStruct<macho::DysymtabLoadCommand>(this, DysymtabLoadCmd);
}

StringRef MachOObjectFile::getStringTableData() const {
  macho::SymtabLoadCommand S = getSymtabLoadCommand();
  return getData().substr(S.StringTableOffset, S.StringTableSize);
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

ObjectFile *ObjectFile::createMachOObjectFile(MemoryBuffer *Buffer) {
  StringRef Magic = Buffer->getBuffer().slice(0, 4);
  error_code ec;
  ObjectFile *Ret;
  if (Magic == "\xFE\xED\xFA\xCE")
    Ret = new MachOObjectFile(Buffer, false, false, ec);
  else if (Magic == "\xCE\xFA\xED\xFE")
    Ret = new MachOObjectFile(Buffer, true, false, ec);
  else if (Magic == "\xFE\xED\xFA\xCF")
    Ret = new MachOObjectFile(Buffer, false, true, ec);
  else if (Magic == "\xCF\xFA\xED\xFE")
    Ret = new MachOObjectFile(Buffer, true, true, ec);
  else
    return NULL;

  if (ec)
    return NULL;
  return Ret;
}

} // end namespace object
} // end namespace llvm
