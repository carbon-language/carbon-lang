//===- MachO.h - MachO object file implementation ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MachOObjectFile class, which binds the MachOObject
// class to the generic ObjectFile wrapper.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_MACHO_H
#define LLVM_OBJECT_MACHO_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Object/MachOFormat.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MachO.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace object {

namespace MachOFormat {
  struct SectionBase {
    char Name[16];
    char SegmentName[16];
  };

  template<bool is64Bits>
  struct Section;

  template<>
  struct Section<false> {
    char Name[16];
    char SegmentName[16];
    support::ulittle32_t Address;
    support::ulittle32_t Size;
    support::ulittle32_t Offset;
    support::ulittle32_t Align;
    support::ulittle32_t RelocationTableOffset;
    support::ulittle32_t NumRelocationTableEntries;
    support::ulittle32_t Flags;
    support::ulittle32_t Reserved1;
    support::ulittle32_t Reserved2;
  };

  template<>
  struct Section<true> {
    char Name[16];
    char SegmentName[16];
    support::ulittle64_t Address;
    support::ulittle64_t Size;
    support::ulittle32_t Offset;
    support::ulittle32_t Align;
    support::ulittle32_t RelocationTableOffset;
    support::ulittle32_t NumRelocationTableEntries;
    support::ulittle32_t Flags;
    support::ulittle32_t Reserved1;
    support::ulittle32_t Reserved2;
    support::ulittle32_t Reserved3;
  };

  struct RelocationEntry {
    support::ulittle32_t Word0;
    support::ulittle32_t Word1;
  };

  struct SymbolTableEntryBase {
    support::ulittle32_t StringIndex;
    uint8_t Type;
    uint8_t SectionIndex;
    support::ulittle16_t Flags;
  };

  template<bool is64Bits>
  struct SymbolTableEntry;

  template<>
  struct SymbolTableEntry<false> {
    support::ulittle32_t StringIndex;
    uint8_t Type;
    uint8_t SectionIndex;
    support::ulittle16_t Flags;
    support::ulittle32_t Value;
  };

  template<>
  struct SymbolTableEntry<true> {
    support::ulittle32_t StringIndex;
    uint8_t Type;
    uint8_t SectionIndex;
    support::ulittle16_t Flags;
    support::ulittle64_t Value;
  };

  struct LoadCommand {
    support::ulittle32_t Type;
    support::ulittle32_t Size;
  };

  struct SymtabLoadCommand {
    support::ulittle32_t Type;
    support::ulittle32_t Size;
    support::ulittle32_t SymbolTableOffset;
    support::ulittle32_t NumSymbolTableEntries;
    support::ulittle32_t StringTableOffset;
    support::ulittle32_t StringTableSize;
  };

  template<bool is64Bits>
  struct SegmentLoadCommand;

  template<>
  struct SegmentLoadCommand<false> {
    support::ulittle32_t Type;
    support::ulittle32_t Size;
    char Name[16];
    support::ulittle32_t VMAddress;
    support::ulittle32_t VMSize;
    support::ulittle32_t FileOffset;
    support::ulittle32_t FileSize;
    support::ulittle32_t MaxVMProtection;
    support::ulittle32_t InitialVMProtection;
    support::ulittle32_t NumSections;
    support::ulittle32_t Flags;
  };

  template<>
  struct SegmentLoadCommand<true> {
    support::ulittle32_t Type;
    support::ulittle32_t Size;
    char Name[16];
    support::ulittle64_t VMAddress;
    support::ulittle64_t VMSize;
    support::ulittle64_t FileOffset;
    support::ulittle64_t FileSize;
    support::ulittle32_t MaxVMProtection;
    support::ulittle32_t InitialVMProtection;
    support::ulittle32_t NumSections;
    support::ulittle32_t Flags;
  };

  struct LinkeditDataLoadCommand {
    support::ulittle32_t Type;
    support::ulittle32_t Size;
    support::ulittle32_t DataOffset;
    support::ulittle32_t DataSize;
  };

  struct Header {
    support::ulittle32_t Magic;
    support::ulittle32_t CPUType;
    support::ulittle32_t CPUSubtype;
    support::ulittle32_t FileType;
    support::ulittle32_t NumLoadCommands;
    support::ulittle32_t SizeOfLoadCommands;
    support::ulittle32_t Flags;
  };
}

class MachOObjectFileBase : public ObjectFile {
public:
  typedef MachOFormat::SymbolTableEntryBase SymbolTableEntryBase;
  typedef MachOFormat::SymtabLoadCommand SymtabLoadCommand;
  typedef MachOFormat::RelocationEntry RelocationEntry;
  typedef MachOFormat::SectionBase SectionBase;
  typedef MachOFormat::LoadCommand LoadCommand;
  typedef MachOFormat::Header Header;

  MachOObjectFileBase(MemoryBuffer *Object, bool Is64Bits, error_code &ec);

  virtual symbol_iterator begin_symbols() const;
  virtual symbol_iterator end_symbols() const;
  virtual symbol_iterator begin_dynamic_symbols() const;
  virtual symbol_iterator end_dynamic_symbols() const;
  virtual library_iterator begin_libraries_needed() const;
  virtual library_iterator end_libraries_needed() const;
  virtual section_iterator end_sections() const;

  virtual uint8_t getBytesInAddress() const;
  virtual StringRef getFileFormatName() const;
  virtual unsigned getArch() const;
  virtual StringRef getLoadName() const;

  // In a MachO file, sections have a segment name. This is used in the .o
  // files. They have a single segment, but this field specifies which segment
  // a section should be put in in the final object.
  StringRef getSectionFinalSegmentName(DataRefImpl Sec) const;

  // Names are stored as 16 bytes. These returns the raw 16 bytes without
  // interpreting them as a C string.
  ArrayRef<char> getSectionRawName(DataRefImpl Sec) const;
  ArrayRef<char>getSectionRawFinalSegmentName(DataRefImpl Sec) const;

  bool is64Bit() const;
  const LoadCommand *getLoadCommandInfo(unsigned Index) const;
  void ReadULEB128s(uint64_t Index, SmallVectorImpl<uint64_t> &Out) const;
  const Header *getHeader() const;
  unsigned getHeaderSize() const;
  StringRef getData(size_t Offset, size_t Size) const;

  static inline bool classof(const Binary *v) {
    return v->isMachO();
  }

protected:
  virtual error_code getSymbolNext(DataRefImpl Symb, SymbolRef &Res) const;
  virtual error_code getSymbolName(DataRefImpl Symb, StringRef &Res) const;
  virtual error_code getSymbolNMTypeChar(DataRefImpl Symb, char &Res) const;
  virtual error_code getSymbolFlags(DataRefImpl Symb, uint32_t &Res) const;
  virtual error_code getSymbolType(DataRefImpl Symb, SymbolRef::Type &Res) const;
  virtual error_code getSymbolSection(DataRefImpl Symb,
                                      section_iterator &Res) const;
  virtual error_code getSymbolValue(DataRefImpl Symb, uint64_t &Val) const;
  virtual error_code getSectionName(DataRefImpl Sec, StringRef &Res) const;
  virtual error_code isSectionData(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionBSS(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionRequiredForExecution(DataRefImpl Sec,
                                                   bool &Res) const;
  virtual error_code isSectionVirtual(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionReadOnlyData(DataRefImpl Sec, bool &Res) const;
  virtual relocation_iterator getSectionRelBegin(DataRefImpl Sec) const;

  virtual error_code getRelocationNext(DataRefImpl Rel,
                                       RelocationRef &Res) const;

  virtual error_code getLibraryNext(DataRefImpl LibData, LibraryRef &Res) const;
  virtual error_code getLibraryPath(DataRefImpl LibData, StringRef &Res) const;

  std::size_t getSectionIndex(DataRefImpl Sec) const;

  typedef SmallVector<DataRefImpl, 1> SectionList;
  SectionList Sections;

  void moveToNextSymbol(DataRefImpl &DRI) const;
  void printRelocationTargetName(const RelocationEntry *RE,
                                 raw_string_ostream &fmt) const;
  const SectionBase *getSectionBase(DataRefImpl DRI) const;
  const SymbolTableEntryBase *getSymbolTableEntryBase(DataRefImpl DRI) const;

private:

  const SymbolTableEntryBase *getSymbolTableEntryBase(DataRefImpl DRI,
                                  const SymtabLoadCommand *SymtabLoadCmd) const;
};

template<bool is64Bits>
struct MachOObjectFileHelperCommon {
  typedef MachOFormat::SegmentLoadCommand<is64Bits> SegmentLoadCommand;
  typedef MachOFormat::SymbolTableEntry<is64Bits> SymbolTableEntry;
  typedef MachOFormat::Section<is64Bits> Section;
};

template<bool is64Bits>
struct MachOObjectFileHelper;

template<>
struct MachOObjectFileHelper<false> :
    public MachOObjectFileHelperCommon<false> {
  static const macho::LoadCommandType SegmentLoadType = macho::LCT_Segment;
};

template<>
struct MachOObjectFileHelper<true> :
    public MachOObjectFileHelperCommon<true> {
  static const macho::LoadCommandType SegmentLoadType = macho::LCT_Segment64;
};

template<bool is64Bits>
class MachOObjectFile : public MachOObjectFileBase {
public:
  static const macho::LoadCommandType SegmentLoadType =
    MachOObjectFileHelper<is64Bits>::SegmentLoadType;
  typedef typename MachOObjectFileHelper<is64Bits>::SegmentLoadCommand
    SegmentLoadCommand;
  typedef typename MachOObjectFileHelper<is64Bits>::SymbolTableEntry
    SymbolTableEntry;
  typedef typename MachOObjectFileHelper<is64Bits>::Section Section;

  MachOObjectFile(MemoryBuffer *Object, error_code &ec) :
    MachOObjectFileBase(Object, is64Bits, ec) {
    DataRefImpl DRI;
    moveToNextSection(DRI);
    uint32_t LoadCommandCount = getHeader()->NumLoadCommands;
    while (DRI.d.a < LoadCommandCount) {
      Sections.push_back(DRI);
      DRI.d.b++;
      moveToNextSection(DRI);
    }
  }

  static inline bool classof(const Binary *v) {
    return v->getType() == getMachOType(true, is64Bits);
  }

  const Section *getSection(DataRefImpl DRI) const;
  const SymbolTableEntry *getSymbolTableEntry(DataRefImpl DRI) const;
  const RelocationEntry *getRelocation(DataRefImpl Rel) const;

  virtual error_code getSectionAddress(DataRefImpl Sec, uint64_t &Res) const;
  virtual error_code getSectionSize(DataRefImpl Sec, uint64_t &Res) const;
  virtual error_code getSectionContents(DataRefImpl Sec, StringRef &Res) const;
  virtual error_code getSectionAlignment(DataRefImpl Sec, uint64_t &Res) const;
  virtual error_code isSectionText(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionZeroInit(DataRefImpl Sec, bool &Res) const;
  virtual relocation_iterator getSectionRelEnd(DataRefImpl Sec) const;
  virtual error_code getRelocationAddress(DataRefImpl Rel, uint64_t &Res) const;
  virtual error_code getRelocationOffset(DataRefImpl Rel, uint64_t &Res) const;
  virtual error_code getRelocationSymbol(DataRefImpl Rel, SymbolRef &Res) const;
  virtual error_code getRelocationAdditionalInfo(DataRefImpl Rel,
                                                 int64_t &Res) const;
  virtual error_code getRelocationType(DataRefImpl Rel, uint64_t &Res) const;
  virtual error_code getRelocationTypeName(DataRefImpl Rel,
                                           SmallVectorImpl<char> &Result) const;
  virtual error_code getRelocationValueString(DataRefImpl Rel,
                                           SmallVectorImpl<char> &Result) const;
  virtual error_code getRelocationHidden(DataRefImpl Rel, bool &Result) const;
  virtual error_code getSymbolFileOffset(DataRefImpl Symb, uint64_t &Res) const;
  virtual error_code sectionContainsSymbol(DataRefImpl Sec, DataRefImpl Symb,
                                           bool &Result) const;
  virtual error_code getSymbolAddress(DataRefImpl Symb, uint64_t &Res) const;
  virtual error_code getSymbolSize(DataRefImpl Symb, uint64_t &Res) const;
  virtual error_code getSectionNext(DataRefImpl Sec, SectionRef &Res) const;
  virtual section_iterator begin_sections() const;
  void moveToNextSection(DataRefImpl &DRI) const;
};

template<bool is64Bits>
const typename MachOObjectFile<is64Bits>::Section *
MachOObjectFile<is64Bits>::getSection(DataRefImpl DRI) const {
  const SectionBase *Addr = getSectionBase(DRI);
  return reinterpret_cast<const Section*>(Addr);
}

template<bool is64Bits>
const typename MachOObjectFile<is64Bits>::SymbolTableEntry *
MachOObjectFile<is64Bits>::getSymbolTableEntry(DataRefImpl DRI) const {
  const SymbolTableEntryBase *Base = getSymbolTableEntryBase(DRI);
  return reinterpret_cast<const SymbolTableEntry*>(Base);
}

template<bool is64Bits>
const typename MachOObjectFile<is64Bits>::RelocationEntry *
MachOObjectFile<is64Bits>::getRelocation(DataRefImpl Rel) const {
  const Section *Sect = getSection(Sections[Rel.d.b]);
  uint32_t RelOffset = Sect->RelocationTableOffset;
  uint64_t Offset = RelOffset + Rel.d.a * sizeof(RelocationEntry);
  StringRef Data = getData(Offset, sizeof(RelocationEntry));
  return reinterpret_cast<const RelocationEntry*>(Data.data());
}

template<bool is64Bits>
error_code
MachOObjectFile<is64Bits>::getSectionAddress(DataRefImpl Sec,
                                             uint64_t &Res) const {
  const Section *Sect = getSection(Sec);
  Res = Sect->Address;
  return object_error::success;
}

template<bool is64Bits>
error_code
MachOObjectFile<is64Bits>::getSectionSize(DataRefImpl Sec,
                                          uint64_t &Res) const {
  const Section *Sect = getSection(Sec);
  Res = Sect->Size;
  return object_error::success;
}

template<bool is64Bits>
error_code
MachOObjectFile<is64Bits>::getSectionContents(DataRefImpl Sec,
                                              StringRef &Res) const {
  const Section *Sect = getSection(Sec);
  Res = getData(Sect->Offset, Sect->Size);
  return object_error::success;
}

template<bool is64Bits>
error_code
MachOObjectFile<is64Bits>::getSectionAlignment(DataRefImpl Sec,
                                               uint64_t &Res) const {
  const Section *Sect = getSection(Sec);
  Res = uint64_t(1) << Sect->Align;
  return object_error::success;
}

template<bool is64Bits>
error_code
MachOObjectFile<is64Bits>::isSectionText(DataRefImpl Sec, bool &Res) const {
  const Section *Sect = getSection(Sec);
  Res = Sect->Flags & macho::SF_PureInstructions;
  return object_error::success;
}

template<bool is64Bits>
error_code
MachOObjectFile<is64Bits>::isSectionZeroInit(DataRefImpl Sec, bool &Res) const {
  const Section *Sect = getSection(Sec);
  unsigned SectionType = Sect->Flags & MachO::SectionFlagMaskSectionType;
  Res = SectionType == MachO::SectionTypeZeroFill ||
    SectionType == MachO::SectionTypeZeroFillLarge;
  return object_error::success;
}

template<bool is64Bits>
relocation_iterator
MachOObjectFile<is64Bits>::getSectionRelEnd(DataRefImpl Sec) const {
  const Section *Sect = getSection(Sec);
  uint32_t LastReloc = Sect->NumRelocationTableEntries;
  DataRefImpl Ret;
  Ret.d.a = LastReloc;
  Ret.d.b = getSectionIndex(Sec);
  return relocation_iterator(RelocationRef(Ret, this));
}

template<bool is64Bits>
error_code
MachOObjectFile<is64Bits>::getRelocationAddress(DataRefImpl Rel,
                                                uint64_t &Res) const {
  const Section *Sect = getSection(Sections[Rel.d.b]);
  uint64_t SectAddress = Sect->Address;
  const RelocationEntry *RE = getRelocation(Rel);
  unsigned Arch = getArch();
  bool isScattered = (Arch != Triple::x86_64) &&
                     (RE->Word0 & macho::RF_Scattered);

  uint64_t RelAddr;
  if (isScattered)
    RelAddr = RE->Word0 & 0xFFFFFF;
  else
    RelAddr = RE->Word0;

  Res = SectAddress + RelAddr;
  return object_error::success;
}

template<bool is64Bits>
error_code
MachOObjectFile<is64Bits>::getRelocationOffset(DataRefImpl Rel,
                                               uint64_t &Res) const {
  const RelocationEntry *RE = getRelocation(Rel);

  unsigned Arch = getArch();
  bool isScattered = (Arch != Triple::x86_64) &&
                     (RE->Word0 & macho::RF_Scattered);
  if (isScattered)
    Res = RE->Word0 & 0xFFFFFF;
  else
    Res = RE->Word0;
  return object_error::success;
}

template<bool is64Bits>
error_code
MachOObjectFile<is64Bits>::getRelocationSymbol(DataRefImpl Rel,
                                               SymbolRef &Res) const {
  const RelocationEntry *RE = getRelocation(Rel);
  uint32_t SymbolIdx = RE->Word1 & 0xffffff;
  bool isExtern = (RE->Word1 >> 27) & 1;

  DataRefImpl Sym;
  moveToNextSymbol(Sym);
  if (isExtern) {
    for (unsigned i = 0; i < SymbolIdx; i++) {
      Sym.d.b++;
      moveToNextSymbol(Sym);
      assert(Sym.d.a < getHeader()->NumLoadCommands &&
             "Relocation symbol index out of range!");
    }
  }
  Res = SymbolRef(Sym, this);
  return object_error::success;
}

template<bool is64Bits>
error_code
MachOObjectFile<is64Bits>::getRelocationAdditionalInfo(DataRefImpl Rel,
                                                       int64_t &Res) const {
  const RelocationEntry *RE = getRelocation(Rel);
  bool isExtern = (RE->Word1 >> 27) & 1;
  Res = 0;
  if (!isExtern) {
    const uint8_t* sectAddress = base();
    const Section *Sect = getSection(Sections[Rel.d.b]);
    sectAddress += Sect->Offset;
    Res = reinterpret_cast<uintptr_t>(sectAddress);
  }
  return object_error::success;
}

template<bool is64Bits>
error_code MachOObjectFile<is64Bits>::getRelocationType(DataRefImpl Rel,
                                                        uint64_t &Res) const {
  const RelocationEntry *RE = getRelocation(Rel);
  Res = RE->Word0;
  Res <<= 32;
  Res |= RE->Word1;
  return object_error::success;
}

template<bool is64Bits>
error_code
MachOObjectFile<is64Bits>::getRelocationTypeName(DataRefImpl Rel,
                                          SmallVectorImpl<char> &Result) const {
    // TODO: Support scattered relocations.
  StringRef res;
  const RelocationEntry *RE = getRelocation(Rel);

  unsigned Arch = getArch();
  bool isScattered = (Arch != Triple::x86_64) &&
                     (RE->Word0 & macho::RF_Scattered);

  unsigned r_type;
  if (isScattered)
    r_type = (RE->Word0 >> 24) & 0xF;
  else
    r_type = (RE->Word1 >> 28) & 0xF;

  switch (Arch) {
    case Triple::x86: {
      static const char *const Table[] =  {
        "GENERIC_RELOC_VANILLA",
        "GENERIC_RELOC_PAIR",
        "GENERIC_RELOC_SECTDIFF",
        "GENERIC_RELOC_PB_LA_PTR",
        "GENERIC_RELOC_LOCAL_SECTDIFF",
        "GENERIC_RELOC_TLV" };

      if (r_type > 6)
        res = "Unknown";
      else
        res = Table[r_type];
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

      if (r_type > 9)
        res = "Unknown";
      else
        res = Table[r_type];
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

      if (r_type > 9)
        res = "Unknown";
      else
        res = Table[r_type];
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

      res = Table[r_type];
      break;
    }
    case Triple::UnknownArch:
      res = "Unknown";
      break;
  }
  Result.append(res.begin(), res.end());
  return object_error::success;
}

template<bool is64Bits>
error_code
MachOObjectFile<is64Bits>::getRelocationValueString(DataRefImpl Rel,
                                          SmallVectorImpl<char> &Result) const {
  const RelocationEntry *RE = getRelocation(Rel);

  unsigned Arch = getArch();
  bool isScattered = (Arch != Triple::x86_64) &&
                     (RE->Word0 & macho::RF_Scattered);

  std::string fmtbuf;
  raw_string_ostream fmt(fmtbuf);

  unsigned Type;
  if (isScattered)
    Type = (RE->Word0 >> 24) & 0xF;
  else
    Type = (RE->Word1 >> 28) & 0xF;

  bool isPCRel;
  if (isScattered)
    isPCRel = ((RE->Word0 >> 30) & 1);
  else
    isPCRel = ((RE->Word1 >> 24) & 1);

  // Determine any addends that should be displayed with the relocation.
  // These require decoding the relocation type, which is triple-specific.

  // X86_64 has entirely custom relocation types.
  if (Arch == Triple::x86_64) {
    bool isPCRel = ((RE->Word1 >> 24) & 1);

    switch (Type) {
      case macho::RIT_X86_64_GOTLoad:   // X86_64_RELOC_GOT_LOAD
      case macho::RIT_X86_64_GOT: {     // X86_64_RELOC_GOT
        printRelocationTargetName(RE, fmt);
        fmt << "@GOT";
        if (isPCRel) fmt << "PCREL";
        break;
      }
      case macho::RIT_X86_64_Subtractor: { // X86_64_RELOC_SUBTRACTOR
        DataRefImpl RelNext = Rel;
        RelNext.d.a++;
        const RelocationEntry *RENext = getRelocation(RelNext);

        // X86_64_SUBTRACTOR must be followed by a relocation of type
        // X86_64_RELOC_UNSIGNED.
        // NOTE: Scattered relocations don't exist on x86_64.
        unsigned RType = (RENext->Word1 >> 28) & 0xF;
        if (RType != 0)
          report_fatal_error("Expected X86_64_RELOC_UNSIGNED after "
                             "X86_64_RELOC_SUBTRACTOR.");

        // The X86_64_RELOC_UNSIGNED contains the minuend symbol,
        // X86_64_SUBTRACTOR contains to the subtrahend.
        printRelocationTargetName(RENext, fmt);
        fmt << "-";
        printRelocationTargetName(RE, fmt);
        break;
      }
      case macho::RIT_X86_64_TLV:
        printRelocationTargetName(RE, fmt);
        fmt << "@TLV";
        if (isPCRel) fmt << "P";
        break;
      case macho::RIT_X86_64_Signed1: // X86_64_RELOC_SIGNED1
        printRelocationTargetName(RE, fmt);
        fmt << "-1";
        break;
      case macho::RIT_X86_64_Signed2: // X86_64_RELOC_SIGNED2
        printRelocationTargetName(RE, fmt);
        fmt << "-2";
        break;
      case macho::RIT_X86_64_Signed4: // X86_64_RELOC_SIGNED4
        printRelocationTargetName(RE, fmt);
        fmt << "-4";
        break;
      default:
        printRelocationTargetName(RE, fmt);
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
        const RelocationEntry *RENext = getRelocation(RelNext);

        // X86 sect diff's must be followed by a relocation of type
        // GENERIC_RELOC_PAIR.
        bool isNextScattered = (Arch != Triple::x86_64) &&
                               (RENext->Word0 & macho::RF_Scattered);
        unsigned RType;
        if (isNextScattered)
          RType = (RENext->Word0 >> 24) & 0xF;
        else
          RType = (RENext->Word1 >> 28) & 0xF;
        if (RType != 1)
          report_fatal_error("Expected GENERIC_RELOC_PAIR after "
                             "GENERIC_RELOC_SECTDIFF.");

        printRelocationTargetName(RE, fmt);
        fmt << "-";
        printRelocationTargetName(RENext, fmt);
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
          const RelocationEntry *RENext = getRelocation(RelNext);

          // X86 sect diff's must be followed by a relocation of type
          // GENERIC_RELOC_PAIR.
          bool isNextScattered = (Arch != Triple::x86_64) &&
                               (RENext->Word0 & macho::RF_Scattered);
          unsigned RType;
          if (isNextScattered)
            RType = (RENext->Word0 >> 24) & 0xF;
          else
            RType = (RENext->Word1 >> 28) & 0xF;
          if (RType != 1)
            report_fatal_error("Expected GENERIC_RELOC_PAIR after "
                               "GENERIC_RELOC_LOCAL_SECTDIFF.");

          printRelocationTargetName(RE, fmt);
          fmt << "-";
          printRelocationTargetName(RENext, fmt);
          break;
        }
        case macho::RIT_Generic_TLV: {
          printRelocationTargetName(RE, fmt);
          fmt << "@TLV";
          if (isPCRel) fmt << "P";
          break;
        }
        default:
          printRelocationTargetName(RE, fmt);
      }
    } else { // ARM-specific relocations
      switch (Type) {
        case macho::RIT_ARM_Half:             // ARM_RELOC_HALF
        case macho::RIT_ARM_HalfDifference: { // ARM_RELOC_HALF_SECTDIFF
          // Half relocations steal a bit from the length field to encode
          // whether this is an upper16 or a lower16 relocation.
          bool isUpper;
          if (isScattered)
            isUpper = (RE->Word0 >> 28) & 1;
          else
            isUpper = (RE->Word1 >> 25) & 1;

          if (isUpper)
            fmt << ":upper16:(";
          else
            fmt << ":lower16:(";
          printRelocationTargetName(RE, fmt);

          DataRefImpl RelNext = Rel;
          RelNext.d.a++;
          const RelocationEntry *RENext = getRelocation(RelNext);

          // ARM half relocs must be followed by a relocation of type
          // ARM_RELOC_PAIR.
          bool isNextScattered = (Arch != Triple::x86_64) &&
                                 (RENext->Word0 & macho::RF_Scattered);
          unsigned RType;
          if (isNextScattered)
            RType = (RENext->Word0 >> 24) & 0xF;
          else
            RType = (RENext->Word1 >> 28) & 0xF;

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
            printRelocationTargetName(RENext, fmt);
          }

          fmt << ")";
          break;
        }
        default: {
          printRelocationTargetName(RE, fmt);
        }
      }
    }
  } else
    printRelocationTargetName(RE, fmt);

  fmt.flush();
  Result.append(fmtbuf.begin(), fmtbuf.end());
  return object_error::success;
}

template<bool is64Bits>
error_code
MachOObjectFile<is64Bits>::getRelocationHidden(DataRefImpl Rel,
                                               bool &Result) const {
  const RelocationEntry *RE = getRelocation(Rel);

  unsigned Arch = getArch();
  bool isScattered = (Arch != Triple::x86_64) &&
                     (RE->Word0 & macho::RF_Scattered);
  unsigned Type;
  if (isScattered)
    Type = (RE->Word0 >> 24) & 0xF;
  else
    Type = (RE->Word1 >> 28) & 0xF;

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
      const RelocationEntry *REPrev = getRelocation(RelPrev);

      unsigned PrevType = (REPrev->Word1 >> 28) & 0xF;

      if (PrevType == macho::RIT_X86_64_Subtractor) Result = true;
    }
  }

  return object_error::success;
}

template<bool is64Bits>
error_code
MachOObjectFile<is64Bits>::getSymbolFileOffset(DataRefImpl Symb,
                                               uint64_t &Res) const {
  const SymbolTableEntry *Entry = getSymbolTableEntry(Symb);
  Res = Entry->Value;
  if (Entry->SectionIndex) {
    const Section *Sec = getSection(Sections[Entry->SectionIndex-1]);
    Res += Sec->Offset - Sec->Address;
  }

  return object_error::success;
}

template<bool is64Bits>
error_code
MachOObjectFile<is64Bits>::sectionContainsSymbol(DataRefImpl Sec,
                                                 DataRefImpl Symb,
                                                 bool &Result) const {
  SymbolRef::Type ST;
  getSymbolType(Symb, ST);
  if (ST == SymbolRef::ST_Unknown) {
    Result = false;
    return object_error::success;
  }

  uint64_t SectBegin, SectEnd;
  getSectionAddress(Sec, SectBegin);
  getSectionSize(Sec, SectEnd);
  SectEnd += SectBegin;

  const SymbolTableEntry *Entry = getSymbolTableEntry(Symb);
  uint64_t SymAddr= Entry->Value;
  Result = (SymAddr >= SectBegin) && (SymAddr < SectEnd);

  return object_error::success;
}

template<bool is64Bits>
error_code MachOObjectFile<is64Bits>::getSymbolAddress(DataRefImpl Symb,
                                                       uint64_t &Res) const {
  const SymbolTableEntry *Entry = getSymbolTableEntry(Symb);
  Res = Entry->Value;
  return object_error::success;
}

template<bool is64Bits>
error_code MachOObjectFile<is64Bits>::getSymbolSize(DataRefImpl DRI,
                                                    uint64_t &Result) const {
  uint32_t LoadCommandCount = getHeader()->NumLoadCommands;
  uint64_t BeginOffset;
  uint64_t EndOffset = 0;
  uint8_t SectionIndex;

  const SymbolTableEntry *Entry = getSymbolTableEntry(DRI);
  BeginOffset = Entry->Value;
  SectionIndex = Entry->SectionIndex;
  if (!SectionIndex) {
    uint32_t flags = SymbolRef::SF_None;
    getSymbolFlags(DRI, flags);
    if (flags & SymbolRef::SF_Common)
      Result = Entry->Value;
    else
      Result = UnknownAddressOrSize;
    return object_error::success;
  }
  // Unfortunately symbols are unsorted so we need to touch all
  // symbols from load command
  DRI.d.b = 0;
  uint32_t Command = DRI.d.a;
  while (Command == DRI.d.a) {
    moveToNextSymbol(DRI);
    if (DRI.d.a < LoadCommandCount) {
      Entry = getSymbolTableEntry(DRI);
      if (Entry->SectionIndex == SectionIndex && Entry->Value > BeginOffset)
        if (!EndOffset || Entry->Value < EndOffset)
          EndOffset = Entry->Value;
    }
    DRI.d.b++;
  }
  if (!EndOffset) {
    uint64_t Size;
    getSectionSize(Sections[SectionIndex-1], Size);
    getSectionAddress(Sections[SectionIndex-1], EndOffset);
    EndOffset += Size;
  }
  Result = EndOffset - BeginOffset;
  return object_error::success;
}

template<bool is64Bits>
error_code MachOObjectFile<is64Bits>::getSectionNext(DataRefImpl Sec,
                                                     SectionRef &Res) const {
  Sec.d.b++;
  moveToNextSection(Sec);
  Res = SectionRef(Sec, this);
  return object_error::success;
}

template<bool is64Bits>
section_iterator MachOObjectFile<is64Bits>::begin_sections() const {
  DataRefImpl DRI;
  moveToNextSection(DRI);
  return section_iterator(SectionRef(DRI, this));
}

template<bool is64Bits>
void MachOObjectFile<is64Bits>::moveToNextSection(DataRefImpl &DRI) const {
  uint32_t LoadCommandCount = getHeader()->NumLoadCommands;
  while (DRI.d.a < LoadCommandCount) {
    const LoadCommand *Command = getLoadCommandInfo(DRI.d.a);
    if (Command->Type == SegmentLoadType) {
      const SegmentLoadCommand *SegmentLoadCmd =
        reinterpret_cast<const SegmentLoadCommand*>(Command);
      if (DRI.d.b < SegmentLoadCmd->NumSections)
        return;
    }

    DRI.d.a++;
    DRI.d.b = 0;
  }
}

}
}

#endif

