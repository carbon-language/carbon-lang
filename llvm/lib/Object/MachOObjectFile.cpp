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
#include "llvm/Object/MachOFormat.h"
#include "llvm/Object/MachOObject.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MachO.h"

#include <cctype>
#include <cstring>
#include <limits>

using namespace llvm;
using namespace object;

namespace llvm {

typedef MachOObject::LoadCommandInfo LoadCommandInfo;

class MachOObjectFile : public ObjectFile {
public:
  MachOObjectFile(MemoryBuffer *Object, MachOObject *MOO, error_code &ec)
    : ObjectFile(Binary::isMachO, Object, ec),
      MachOObj(MOO),
      RegisteredStringTable(std::numeric_limits<uint32_t>::max()) {}

  virtual symbol_iterator begin_symbols() const;
  virtual symbol_iterator end_symbols() const;
  virtual section_iterator begin_sections() const;
  virtual section_iterator end_sections() const;

  virtual uint8_t getBytesInAddress() const;
  virtual StringRef getFileFormatName() const;
  virtual unsigned getArch() const;

protected:
  virtual error_code getSymbolNext(DataRefImpl Symb, SymbolRef &Res) const;
  virtual error_code getSymbolName(DataRefImpl Symb, StringRef &Res) const;
  virtual error_code getSymbolAddress(DataRefImpl Symb, uint64_t &Res) const;
  virtual error_code getSymbolSize(DataRefImpl Symb, uint64_t &Res) const;
  virtual error_code getSymbolNMTypeChar(DataRefImpl Symb, char &Res) const;
  virtual error_code isSymbolInternal(DataRefImpl Symb, bool &Res) const;

  virtual error_code getSectionNext(DataRefImpl Sec, SectionRef &Res) const;
  virtual error_code getSectionName(DataRefImpl Sec, StringRef &Res) const;
  virtual error_code getSectionAddress(DataRefImpl Sec, uint64_t &Res) const;
  virtual error_code getSectionSize(DataRefImpl Sec, uint64_t &Res) const;
  virtual error_code getSectionContents(DataRefImpl Sec, StringRef &Res) const;
  virtual error_code isSectionText(DataRefImpl Sec, bool &Res) const;

private:
  MachOObject *MachOObj;
  mutable uint32_t RegisteredStringTable;

  void moveToNextSection(DataRefImpl &DRI) const;
  void getSymbolTableEntry(DataRefImpl DRI,
                           InMemoryStruct<macho::SymbolTableEntry> &Res) const;
  void moveToNextSymbol(DataRefImpl &DRI) const;
  void getSection(DataRefImpl DRI, InMemoryStruct<macho::Section> &Res) const;
  void getSection64(DataRefImpl DRI,
                    InMemoryStruct<macho::Section64> &Res) const;
};

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


error_code MachOObjectFile::getSymbolNext(DataRefImpl DRI,
                                          SymbolRef &Result) const {
  DRI.d.b++;
  moveToNextSymbol(DRI);
  Result = SymbolRef(DRI, this);
  return object_error::success;
}

error_code MachOObjectFile::getSymbolName(DataRefImpl DRI,
                                          StringRef &Result) const {
  InMemoryStruct<macho::SymbolTableEntry> Entry;
  getSymbolTableEntry(DRI, Entry);
  Result = MachOObj->getStringAtIndex(Entry->StringIndex);
  return object_error::success;
}

error_code MachOObjectFile::getSymbolAddress(DataRefImpl DRI,
                                             uint64_t &Result) const {
  InMemoryStruct<macho::SymbolTableEntry> Entry;
  getSymbolTableEntry(DRI, Entry);
  Result = Entry->Value;
  return object_error::success;
}

error_code MachOObjectFile::getSymbolSize(DataRefImpl DRI,
                                          uint64_t &Result) const {
  Result = UnknownAddressOrSize;
  return object_error::success;
}

error_code MachOObjectFile::getSymbolNMTypeChar(DataRefImpl DRI,
                                                char &Result) const {
  InMemoryStruct<macho::SymbolTableEntry> Entry;
  getSymbolTableEntry(DRI, Entry);

  char Char;
  switch (Entry->Type & macho::STF_TypeMask) {
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

  if (Entry->Flags & (macho::STF_External | macho::STF_PrivateExtern))
    Char = toupper(Char);
  Result = Char;
  return object_error::success;
}

error_code MachOObjectFile::isSymbolInternal(DataRefImpl DRI,
                                             bool &Result) const {
  InMemoryStruct<macho::SymbolTableEntry> Entry;
  getSymbolTableEntry(DRI, Entry);
  Result = Entry->Flags & macho::STF_StabsEntryMask;
  return object_error::success;
}

ObjectFile::symbol_iterator MachOObjectFile::begin_symbols() const {
  // DRI.d.a = segment number; DRI.d.b = symbol index.
  DataRefImpl DRI;
  DRI.d.a = DRI.d.b = 0;
  moveToNextSymbol(DRI);
  return symbol_iterator(SymbolRef(DRI, this));
}

ObjectFile::symbol_iterator MachOObjectFile::end_symbols() const {
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

    strcpy(result, SLC->Name);
    strcat(result, ",");
    strcat(result, Sect->Name);
  } else {
    InMemoryStruct<macho::SegmentLoadCommand> SLC;
    LoadCommandInfo LCI = MachOObj->getLoadCommandInfo(DRI.d.a);
    MachOObj->ReadSegmentLoadCommand(LCI, SLC);
    InMemoryStruct<macho::Section> Sect;
    MachOObj->ReadSection(LCI, DRI.d.b, Sect);

    strcpy(result, SLC->Name);
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

ObjectFile::section_iterator MachOObjectFile::begin_sections() const {
  DataRefImpl DRI;
  DRI.d.a = DRI.d.b = 0;
  moveToNextSection(DRI);
  return section_iterator(SectionRef(DRI, this));
}

ObjectFile::section_iterator MachOObjectFile::end_sections() const {
  DataRefImpl DRI;
  DRI.d.a = MachOObj->getHeader().NumLoadCommands;
  DRI.d.b = 0;
  return section_iterator(SectionRef(DRI, this));
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

} // end namespace llvm

