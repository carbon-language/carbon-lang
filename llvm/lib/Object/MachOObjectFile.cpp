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

#include "llvm/ADT/Triple.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/MachOFormat.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cctype>
#include <cstring>
#include <limits>

using namespace llvm;
using namespace object;

namespace llvm {
namespace object {

MachOObjectFile::MachOObjectFile(MemoryBuffer *Object, MachOObject *MOO,
                                 error_code &ec)
    : ObjectFile(Binary::isMachO, Object, ec),
      MachOObj(MOO),
      RegisteredStringTable(std::numeric_limits<uint32_t>::max()) {
  DataRefImpl DRI;
  DRI.d.a = DRI.d.b = 0;
  moveToNextSection(DRI);
  uint32_t LoadCommandCount = MachOObj->getHeader().NumLoadCommands;
  while (DRI.d.a < LoadCommandCount) {
    Sections.push_back(DRI);
    DRI.d.b++;
    moveToNextSection(DRI);
  }
}


ObjectFile *ObjectFile::createMachOObjectFile(MemoryBuffer *Buffer) {
  error_code ec;
  std::string Err;
  MachOObject *MachOObj = MachOObject::LoadFromBuffer(Buffer, &Err);
  if (!MachOObj)
    return NULL;
  return new MachOObjectFile(Buffer, MachOObj, ec);
}

/*===-- Symbols -----------------------------------------------------------===*/

void MachOObjectFile::moveToNextSymbol(DataRefImpl &DRI) const {
  uint32_t LoadCommandCount = MachOObj->getHeader().NumLoadCommands;
  while (DRI.d.a < LoadCommandCount) {
    LoadCommandInfo LCI = MachOObj->getLoadCommandInfo(DRI.d.a);
    if (LCI.Command.Type == macho::LCT_Symtab) {
      InMemoryStruct<macho::SymtabLoadCommand> SymtabLoadCmd;
      MachOObj->ReadSymtabLoadCommand(LCI, SymtabLoadCmd);
      if (DRI.d.b < SymtabLoadCmd->NumSymbolTableEntries)
        return;
    }

    DRI.d.a++;
    DRI.d.b = 0;
  }
}

void MachOObjectFile::getSymbolTableEntry(DataRefImpl DRI,
    InMemoryStruct<macho::SymbolTableEntry> &Res) const {
  InMemoryStruct<macho::SymtabLoadCommand> SymtabLoadCmd;
  LoadCommandInfo LCI = MachOObj->getLoadCommandInfo(DRI.d.a);
  MachOObj->ReadSymtabLoadCommand(LCI, SymtabLoadCmd);

  if (RegisteredStringTable != DRI.d.a) {
    MachOObj->RegisterStringTable(*SymtabLoadCmd);
    RegisteredStringTable = DRI.d.a;
  }

  MachOObj->ReadSymbolTableEntry(SymtabLoadCmd->SymbolTableOffset, DRI.d.b,
                                 Res);
}

void MachOObjectFile::getSymbol64TableEntry(DataRefImpl DRI,
    InMemoryStruct<macho::Symbol64TableEntry> &Res) const {
  InMemoryStruct<macho::SymtabLoadCommand> SymtabLoadCmd;
  LoadCommandInfo LCI = MachOObj->getLoadCommandInfo(DRI.d.a);
  MachOObj->ReadSymtabLoadCommand(LCI, SymtabLoadCmd);

  if (RegisteredStringTable != DRI.d.a) {
    MachOObj->RegisterStringTable(*SymtabLoadCmd);
    RegisteredStringTable = DRI.d.a;
  }

  MachOObj->ReadSymbol64TableEntry(SymtabLoadCmd->SymbolTableOffset, DRI.d.b,
                                   Res);
}


error_code MachOObjectFile::getSymbolNext(DataRefImpl DRI,
                                          SymbolRef &Result) const {
  DRI.d.b++;
  moveToNextSymbol(DRI);
  Result = SymbolRef(DRI, this);
  return object_error::success;
}

error_code MachOObjectFile::getSymbolName(DataRefImpl DRI,
                                          StringRef &Result) const {
  if (MachOObj->is64Bit()) {
    InMemoryStruct<macho::Symbol64TableEntry> Entry;
    getSymbol64TableEntry(DRI, Entry);
    Result = MachOObj->getStringAtIndex(Entry->StringIndex);
  } else {
    InMemoryStruct<macho::SymbolTableEntry> Entry;
    getSymbolTableEntry(DRI, Entry);
    Result = MachOObj->getStringAtIndex(Entry->StringIndex);
  }
  return object_error::success;
}

error_code MachOObjectFile::getSymbolOffset(DataRefImpl DRI,
                                             uint64_t &Result) const {
  if (MachOObj->is64Bit()) {
    InMemoryStruct<macho::Symbol64TableEntry> Entry;
    getSymbol64TableEntry(DRI, Entry);
    Result = Entry->Value;
  } else {
    InMemoryStruct<macho::SymbolTableEntry> Entry;
    getSymbolTableEntry(DRI, Entry);
    Result = Entry->Value;
  }
  return object_error::success;
}

error_code MachOObjectFile::getSymbolAddress(DataRefImpl DRI,
                                             uint64_t &Result) const {
  uint64_t SymbolOffset;
  uint8_t SectionIndex;
  if (MachOObj->is64Bit()) {
    InMemoryStruct<macho::Symbol64TableEntry> Entry;
    getSymbol64TableEntry(DRI, Entry);
    SymbolOffset = Entry->Value;
    SectionIndex = Entry->SectionIndex;
  } else {
    InMemoryStruct<macho::SymbolTableEntry> Entry;
    getSymbolTableEntry(DRI, Entry);
    SymbolOffset = Entry->Value;
    SectionIndex = Entry->SectionIndex;
  }
  getSectionAddress(Sections[SectionIndex-1], Result);
  Result += SymbolOffset;

  return object_error::success;
}

error_code MachOObjectFile::getSymbolSize(DataRefImpl DRI,
                                          uint64_t &Result) const {
  Result = UnknownAddressOrSize;
  return object_error::success;
}

error_code MachOObjectFile::getSymbolNMTypeChar(DataRefImpl DRI,
                                                char &Result) const {
  uint8_t Type, Flags;
  if (MachOObj->is64Bit()) {
    InMemoryStruct<macho::Symbol64TableEntry> Entry;
    getSymbol64TableEntry(DRI, Entry);
    Type = Entry->Type;
    Flags = Entry->Flags;
  } else {
    InMemoryStruct<macho::SymbolTableEntry> Entry;
    getSymbolTableEntry(DRI, Entry);
    Type = Entry->Type;
    Flags = Entry->Flags;
  }

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
    Char = toupper(Char);
  Result = Char;
  return object_error::success;
}

error_code MachOObjectFile::isSymbolInternal(DataRefImpl DRI,
                                             bool &Result) const {
  if (MachOObj->is64Bit()) {
    InMemoryStruct<macho::Symbol64TableEntry> Entry;
    getSymbol64TableEntry(DRI, Entry);
    Result = Entry->Flags & macho::STF_StabsEntryMask;
  } else {
    InMemoryStruct<macho::SymbolTableEntry> Entry;
    getSymbolTableEntry(DRI, Entry);
    Result = Entry->Flags & macho::STF_StabsEntryMask;
  }
  return object_error::success;
}

error_code MachOObjectFile::isSymbolGlobal(DataRefImpl Symb, bool &Res) const {

  if (MachOObj->is64Bit()) {
    InMemoryStruct<macho::Symbol64TableEntry> Entry;
    getSymbol64TableEntry(Symb, Entry);
    Res = Entry->Type & MachO::NlistMaskExternal;
  } else {
    InMemoryStruct<macho::SymbolTableEntry> Entry;
    getSymbolTableEntry(Symb, Entry);
    Res = Entry->Type & MachO::NlistMaskExternal;
  }
  return object_error::success;
}

error_code MachOObjectFile::getSymbolType(DataRefImpl Symb,
                                          SymbolRef::SymbolType &Res) const {
  uint8_t n_type;
  if (MachOObj->is64Bit()) {
    InMemoryStruct<macho::Symbol64TableEntry> Entry;
    getSymbol64TableEntry(Symb, Entry);
    n_type = Entry->Type;
  } else {
    InMemoryStruct<macho::SymbolTableEntry> Entry;
    getSymbolTableEntry(Symb, Entry);
    n_type = Entry->Type;
  }
  Res = SymbolRef::ST_Other;

  // If this is a STAB debugging symbol, we can do nothing more.
  if (n_type & MachO::NlistMaskStab)
    return object_error::success;

  switch (n_type & MachO::NlistMaskType) {
    case MachO::NListTypeUndefined :
      Res = SymbolRef::ST_External;
      break;
    case MachO::NListTypeSection :
      Res = SymbolRef::ST_Function;
      break;
  }
  return object_error::success;
}


symbol_iterator MachOObjectFile::begin_symbols() const {
  // DRI.d.a = segment number; DRI.d.b = symbol index.
  DataRefImpl DRI;
  DRI.d.a = DRI.d.b = 0;
  moveToNextSymbol(DRI);
  return symbol_iterator(SymbolRef(DRI, this));
}

symbol_iterator MachOObjectFile::end_symbols() const {
  DataRefImpl DRI;
  DRI.d.a = MachOObj->getHeader().NumLoadCommands;
  DRI.d.b = 0;
  return symbol_iterator(SymbolRef(DRI, this));
}


/*===-- Sections ----------------------------------------------------------===*/

void MachOObjectFile::moveToNextSection(DataRefImpl &DRI) const {
  uint32_t LoadCommandCount = MachOObj->getHeader().NumLoadCommands;
  while (DRI.d.a < LoadCommandCount) {
    LoadCommandInfo LCI = MachOObj->getLoadCommandInfo(DRI.d.a);
    if (LCI.Command.Type == macho::LCT_Segment) {
      InMemoryStruct<macho::SegmentLoadCommand> SegmentLoadCmd;
      MachOObj->ReadSegmentLoadCommand(LCI, SegmentLoadCmd);
      if (DRI.d.b < SegmentLoadCmd->NumSections)
        return;
    } else if (LCI.Command.Type == macho::LCT_Segment64) {
      InMemoryStruct<macho::Segment64LoadCommand> Segment64LoadCmd;
      MachOObj->ReadSegment64LoadCommand(LCI, Segment64LoadCmd);
      if (DRI.d.b < Segment64LoadCmd->NumSections)
        return;
    }

    DRI.d.a++;
    DRI.d.b = 0;
  }
}

error_code MachOObjectFile::getSectionNext(DataRefImpl DRI,
                                           SectionRef &Result) const {
  DRI.d.b++;
  moveToNextSection(DRI);
  Result = SectionRef(DRI, this);
  return object_error::success;
}

void
MachOObjectFile::getSection(DataRefImpl DRI,
                            InMemoryStruct<macho::Section> &Res) const {
  InMemoryStruct<macho::SegmentLoadCommand> SLC;
  LoadCommandInfo LCI = MachOObj->getLoadCommandInfo(DRI.d.a);
  MachOObj->ReadSegmentLoadCommand(LCI, SLC);
  MachOObj->ReadSection(LCI, DRI.d.b, Res);
}

std::size_t MachOObjectFile::getSectionIndex(DataRefImpl Sec) const {
  SectionList::const_iterator loc =
    std::find(Sections.begin(), Sections.end(), Sec);
  assert(loc != Sections.end() && "Sec is not a valid section!");
  return std::distance(Sections.begin(), loc);
}

void
MachOObjectFile::getSection64(DataRefImpl DRI,
                            InMemoryStruct<macho::Section64> &Res) const {
  InMemoryStruct<macho::Segment64LoadCommand> SLC;
  LoadCommandInfo LCI = MachOObj->getLoadCommandInfo(DRI.d.a);
  MachOObj->ReadSegment64LoadCommand(LCI, SLC);
  MachOObj->ReadSection64(LCI, DRI.d.b, Res);
}

static bool is64BitLoadCommand(const MachOObject *MachOObj, DataRefImpl DRI) {
  LoadCommandInfo LCI = MachOObj->getLoadCommandInfo(DRI.d.a);
  if (LCI.Command.Type == macho::LCT_Segment64)
    return true;
  assert(LCI.Command.Type == macho::LCT_Segment && "Unexpected Type.");
  return false;
}

error_code MachOObjectFile::getSectionName(DataRefImpl DRI,
                                           StringRef &Result) const {
  // FIXME: thread safety.
  static char result[34];
  if (is64BitLoadCommand(MachOObj, DRI)) {
    InMemoryStruct<macho::Segment64LoadCommand> SLC;
    LoadCommandInfo LCI = MachOObj->getLoadCommandInfo(DRI.d.a);
    MachOObj->ReadSegment64LoadCommand(LCI, SLC);
    InMemoryStruct<macho::Section64> Sect;
    MachOObj->ReadSection64(LCI, DRI.d.b, Sect);

    strcpy(result, Sect->SegmentName);
    strcat(result, ",");
    strcat(result, Sect->Name);
  } else {
    InMemoryStruct<macho::SegmentLoadCommand> SLC;
    LoadCommandInfo LCI = MachOObj->getLoadCommandInfo(DRI.d.a);
    MachOObj->ReadSegmentLoadCommand(LCI, SLC);
    InMemoryStruct<macho::Section> Sect;
    MachOObj->ReadSection(LCI, DRI.d.b, Sect);

    strcpy(result, Sect->SegmentName);
    strcat(result, ",");
    strcat(result, Sect->Name);
  }
  Result = StringRef(result);
  return object_error::success;
}

error_code MachOObjectFile::getSectionAddress(DataRefImpl DRI,
                                              uint64_t &Result) const {
  if (is64BitLoadCommand(MachOObj, DRI)) {
    InMemoryStruct<macho::Section64> Sect;
    getSection64(DRI, Sect);
    Result = Sect->Address;
  } else {
    InMemoryStruct<macho::Section> Sect;
    getSection(DRI, Sect);
    Result = Sect->Address;
  }
  return object_error::success;
}

error_code MachOObjectFile::getSectionSize(DataRefImpl DRI,
                                           uint64_t &Result) const {
  if (is64BitLoadCommand(MachOObj, DRI)) {
    InMemoryStruct<macho::Section64> Sect;
    getSection64(DRI, Sect);
    Result = Sect->Size;
  } else {
    InMemoryStruct<macho::Section> Sect;
    getSection(DRI, Sect);
    Result = Sect->Size;
  }
  return object_error::success;
}

error_code MachOObjectFile::getSectionContents(DataRefImpl DRI,
                                               StringRef &Result) const {
  if (is64BitLoadCommand(MachOObj, DRI)) {
    InMemoryStruct<macho::Section64> Sect;
    getSection64(DRI, Sect);
    Result = MachOObj->getData(Sect->Offset, Sect->Size);
  } else {
    InMemoryStruct<macho::Section> Sect;
    getSection(DRI, Sect);
    Result = MachOObj->getData(Sect->Offset, Sect->Size);
  }
  return object_error::success;
}

error_code MachOObjectFile::getSectionAlignment(DataRefImpl DRI,
                                                uint64_t &Result) const {
  if (is64BitLoadCommand(MachOObj, DRI)) {
    InMemoryStruct<macho::Section64> Sect;
    getSection64(DRI, Sect);
    Result = uint64_t(1) << Sect->Align;
  } else {
    InMemoryStruct<macho::Section> Sect;
    getSection(DRI, Sect);
    Result = uint64_t(1) << Sect->Align;
  }
  return object_error::success;
}

error_code MachOObjectFile::isSectionText(DataRefImpl DRI,
                                          bool &Result) const {
  if (is64BitLoadCommand(MachOObj, DRI)) {
    InMemoryStruct<macho::Section64> Sect;
    getSection64(DRI, Sect);
    Result = !strcmp(Sect->Name, "__text");
  } else {
    InMemoryStruct<macho::Section> Sect;
    getSection(DRI, Sect);
    Result = !strcmp(Sect->Name, "__text");
  }
  return object_error::success;
}

error_code MachOObjectFile::isSectionData(DataRefImpl DRI,
                                          bool &Result) const {
  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

error_code MachOObjectFile::isSectionBSS(DataRefImpl DRI,
                                         bool &Result) const {
  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

error_code MachOObjectFile::sectionContainsSymbol(DataRefImpl Sec,
                                                  DataRefImpl Symb,
                                                  bool &Result) const {
  SymbolRef::SymbolType ST;
  getSymbolType(Symb, ST);
  if (ST == SymbolRef::ST_External) {
    Result = false;
    return object_error::success;
  }

  uint64_t SectBegin, SectEnd;
  getSectionAddress(Sec, SectBegin);
  getSectionSize(Sec, SectEnd);
  SectEnd += SectBegin;

  if (MachOObj->is64Bit()) {
    InMemoryStruct<macho::Symbol64TableEntry> Entry;
    getSymbol64TableEntry(Symb, Entry);
    uint64_t SymAddr= Entry->Value;
    Result = (SymAddr >= SectBegin) && (SymAddr < SectEnd);
  } else {
    InMemoryStruct<macho::SymbolTableEntry> Entry;
    getSymbolTableEntry(Symb, Entry);
    uint64_t SymAddr= Entry->Value;
    Result = (SymAddr >= SectBegin) && (SymAddr < SectEnd);
  }

  return object_error::success;
}

relocation_iterator MachOObjectFile::getSectionRelBegin(DataRefImpl Sec) const {
  DataRefImpl ret;
  ret.d.a = 0;
  ret.d.b = getSectionIndex(Sec);
  return relocation_iterator(RelocationRef(ret, this));
}
relocation_iterator MachOObjectFile::getSectionRelEnd(DataRefImpl Sec) const {
  uint32_t last_reloc;
  if (is64BitLoadCommand(MachOObj, Sec)) {
    InMemoryStruct<macho::Section64> Sect;
    getSection64(Sec, Sect);
    last_reloc = Sect->NumRelocationTableEntries;
  } else {
    InMemoryStruct<macho::Section> Sect;
    getSection(Sec, Sect);
    last_reloc = Sect->NumRelocationTableEntries;
  }
  DataRefImpl ret;
  ret.d.a = last_reloc;
  ret.d.b = getSectionIndex(Sec);
  return relocation_iterator(RelocationRef(ret, this));
}

section_iterator MachOObjectFile::begin_sections() const {
  DataRefImpl DRI;
  DRI.d.a = DRI.d.b = 0;
  moveToNextSection(DRI);
  return section_iterator(SectionRef(DRI, this));
}

section_iterator MachOObjectFile::end_sections() const {
  DataRefImpl DRI;
  DRI.d.a = MachOObj->getHeader().NumLoadCommands;
  DRI.d.b = 0;
  return section_iterator(SectionRef(DRI, this));
}

/*===-- Relocations -------------------------------------------------------===*/

void MachOObjectFile::
getRelocation(DataRefImpl Rel,
              InMemoryStruct<macho::RelocationEntry> &Res) const {
  uint32_t relOffset;
  if (MachOObj->is64Bit()) {
    InMemoryStruct<macho::Section64> Sect;
    getSection64(Sections[Rel.d.b], Sect);
    relOffset = Sect->RelocationTableOffset;
  } else {
    InMemoryStruct<macho::Section> Sect;
    getSection(Sections[Rel.d.b], Sect);
    relOffset = Sect->RelocationTableOffset;
  }
  MachOObj->ReadRelocationEntry(relOffset, Rel.d.a, Res);
}
error_code MachOObjectFile::getRelocationNext(DataRefImpl Rel,
                                              RelocationRef &Res) const {
  ++Rel.d.a;
  Res = RelocationRef(Rel, this);
  return object_error::success;
}
error_code MachOObjectFile::getRelocationAddress(DataRefImpl Rel,
                                                 uint64_t &Res) const {
  const uint8_t* sectAddress = base();
  if (MachOObj->is64Bit()) {
    InMemoryStruct<macho::Section64> Sect;
    getSection64(Sections[Rel.d.b], Sect);
    sectAddress += Sect->Offset;
  } else {
    InMemoryStruct<macho::Section> Sect;
    getSection(Sections[Rel.d.b], Sect);
    sectAddress += Sect->Offset;
  }
  InMemoryStruct<macho::RelocationEntry> RE;
  getRelocation(Rel, RE);
  Res = reinterpret_cast<uintptr_t>(sectAddress + RE->Word0);
  return object_error::success;
}
error_code MachOObjectFile::getRelocationSymbol(DataRefImpl Rel,
                                                SymbolRef &Res) const {
  InMemoryStruct<macho::RelocationEntry> RE;
  getRelocation(Rel, RE);
  uint32_t SymbolIdx = RE->Word1 & 0xffffff;
  bool isExtern = (RE->Word1 >> 27) & 1;

  DataRefImpl Sym;
  Sym.d.a = Sym.d.b = 0;
  moveToNextSymbol(Sym);
  if (isExtern) {
    for (unsigned i = 0; i < SymbolIdx; i++) {
      Sym.d.b++;
      moveToNextSymbol(Sym);
      assert(Sym.d.a < MachOObj->getHeader().NumLoadCommands &&
             "Relocation symbol index out of range!");
    }
  }
  Res = SymbolRef(Sym, this);
  return object_error::success;
}
error_code MachOObjectFile::getRelocationType(DataRefImpl Rel,
                                              uint32_t &Res) const {
  InMemoryStruct<macho::RelocationEntry> RE;
  getRelocation(Rel, RE);
  Res = RE->Word1;
  return object_error::success;
}
error_code MachOObjectFile::getRelocationTypeName(DataRefImpl Rel,
                                          SmallVectorImpl<char> &Result) const {
  return object_error::success;
}
error_code MachOObjectFile::getRelocationAdditionalInfo(DataRefImpl Rel,
                                                        int64_t &Res) const {
  InMemoryStruct<macho::RelocationEntry> RE;
  getRelocation(Rel, RE);
  bool isExtern = (RE->Word1 >> 27) & 1;
  Res = 0;
  if (!isExtern) {
    const uint8_t* sectAddress = base();
    if (MachOObj->is64Bit()) {
      InMemoryStruct<macho::Section64> Sect;
      getSection64(Sections[Rel.d.b], Sect);
      sectAddress += Sect->Offset;
    } else {
      InMemoryStruct<macho::Section> Sect;
      getSection(Sections[Rel.d.b], Sect);
      sectAddress += Sect->Offset;
    }
    Res = reinterpret_cast<uintptr_t>(sectAddress);
  }
  return object_error::success;
}
error_code MachOObjectFile::getRelocationValueString(DataRefImpl Rel,
                                          SmallVectorImpl<char> &Result) const {
  return object_error::success;
}

/*===-- Miscellaneous -----------------------------------------------------===*/

uint8_t MachOObjectFile::getBytesInAddress() const {
  return MachOObj->is64Bit() ? 8 : 4;
}

StringRef MachOObjectFile::getFileFormatName() const {
  if (!MachOObj->is64Bit()) {
    switch (MachOObj->getHeader().CPUType) {
    case llvm::MachO::CPUTypeI386:
      return "Mach-O 32-bit i386";
    case llvm::MachO::CPUTypeARM:
      return "Mach-O arm";
    case llvm::MachO::CPUTypePowerPC:
      return "Mach-O 32-bit ppc";
    default:
      assert((MachOObj->getHeader().CPUType & llvm::MachO::CPUArchABI64) == 0 &&
             "64-bit object file when we're not 64-bit?");
      return "Mach-O 32-bit unknown";
    }
  }

  switch (MachOObj->getHeader().CPUType) {
  case llvm::MachO::CPUTypeX86_64:
    return "Mach-O 64-bit x86-64";
  case llvm::MachO::CPUTypePowerPC64:
    return "Mach-O 64-bit ppc64";
  default:
    assert((MachOObj->getHeader().CPUType & llvm::MachO::CPUArchABI64) == 1 &&
           "32-bit object file when we're 64-bit?");
    return "Mach-O 64-bit unknown";
  }
}

unsigned MachOObjectFile::getArch() const {
  switch (MachOObj->getHeader().CPUType) {
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

} // end namespace object
} // end namespace llvm
