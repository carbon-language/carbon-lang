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
#include "llvm/Support/Casting.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cctype>
#include <cstring>
#include <limits>

using namespace llvm;
using namespace object;

namespace llvm {
namespace object {

MachOObjectFileBase::MachOObjectFileBase(MemoryBuffer *Object, bool Is64bits,
                                         error_code &ec)
    : ObjectFile(getMachOType(true, Is64bits), Object) {
}

bool MachOObjectFileBase::is64Bit() const {
  return isa<MachOObjectFile64Le>(this);
}

const MachOObjectFileBase::LoadCommand *
MachOObjectFileBase::getLoadCommandInfo(unsigned Index) const {
  uint64_t Offset;
  uint64_t NewOffset = getHeaderSize();
  const LoadCommand *Load;
  unsigned I = 0;
  do {
    Offset = NewOffset;
    StringRef Data = getData(Offset, sizeof(LoadCommand));
    Load = reinterpret_cast<const LoadCommand*>(Data.data());
    NewOffset = Offset + Load->Size;
    ++I;
  } while (I != Index + 1);

  return Load;
}

void MachOObjectFileBase::ReadULEB128s(uint64_t Index,
                                       SmallVectorImpl<uint64_t> &Out) const {
  DataExtractor extractor(ObjectFile::getData(), true, 0);

  uint32_t offset = Index;
  uint64_t data = 0;
  while (uint64_t delta = extractor.getULEB128(&offset)) {
    data += delta;
    Out.push_back(data);
  }
}

const MachOObjectFileBase::Header *MachOObjectFileBase::getHeader() const {
  StringRef Data = getData(0, sizeof(Header));
  return reinterpret_cast<const Header*>(Data.data());
}

unsigned MachOObjectFileBase::getHeaderSize() const {
  return is64Bit() ? macho::Header64Size : macho::Header32Size;
}

StringRef MachOObjectFileBase::getData(size_t Offset, size_t Size) const {
  return ObjectFile::getData().substr(Offset, Size);
}

const MachOObjectFileBase::RelocationEntry *
MachOObjectFileBase::getRelocation(DataRefImpl Rel) const {
  if (const MachOObjectFile32Le *O = dyn_cast<MachOObjectFile32Le>(this))
    return O->getRelocation(Rel);
  const MachOObjectFile64Le *O = dyn_cast<MachOObjectFile64Le>(this);
  return O->getRelocation(Rel);
}

bool MachOObjectFileBase::isScattered(const RelocationEntry *RE) const {
  unsigned Arch = getArch();
  return (Arch != Triple::x86_64) && (RE->Address & macho::RF_Scattered);
}

bool MachOObjectFileBase::isPCRel(const RelocationEntry *RE) const {
  if (isScattered(RE)) {
    const ScatteredRelocationEntry *SRE =
      reinterpret_cast<const ScatteredRelocationEntry *>(RE);
    return SRE->PCRel;
  }
  return RE->PCRel;
}

unsigned MachOObjectFileBase::getLength(const RelocationEntry *RE) const {
  if (isScattered(RE)) {
    const ScatteredRelocationEntry *SRE =
      reinterpret_cast<const ScatteredRelocationEntry *>(RE);
    return SRE->Length;
  }
  return RE->Length;
}

unsigned MachOObjectFileBase::getType(const RelocationEntry *RE) const {
  if (isScattered(RE)) {
    const ScatteredRelocationEntry *SRE =
      reinterpret_cast<const ScatteredRelocationEntry *>(RE);
    return SRE->Type;
  }
  return RE->Type;
}

ObjectFile *ObjectFile::createMachOObjectFile(MemoryBuffer *Buffer) {
  StringRef Magic = Buffer->getBuffer().slice(0, 4);
  error_code ec;
  bool Is64Bits = Magic == "\xFE\xED\xFA\xCF" || Magic == "\xCF\xFA\xED\xFE";
  ObjectFile *Ret;
  if (Is64Bits)
    Ret = new MachOObjectFile64Le(Buffer, ec);
  else
    Ret = new MachOObjectFile32Le(Buffer, ec);
  if (ec)
    return NULL;
  return Ret;
}

/*===-- Symbols -----------------------------------------------------------===*/

void MachOObjectFileBase::moveToNextSymbol(DataRefImpl &DRI) const {
  uint32_t LoadCommandCount = getHeader()->NumLoadCommands;
  while (DRI.d.a < LoadCommandCount) {
    const LoadCommand *Command = getLoadCommandInfo(DRI.d.a);
    if (Command->Type == macho::LCT_Symtab) {
      const SymtabLoadCommand *SymtabLoadCmd =
        reinterpret_cast<const SymtabLoadCommand*>(Command);
      if (DRI.d.b < SymtabLoadCmd->NumSymbolTableEntries)
        return;
    }

    DRI.d.a++;
    DRI.d.b = 0;
  }
}

const MachOObjectFileBase::SymbolTableEntryBase *
MachOObjectFileBase::getSymbolTableEntryBase(DataRefImpl DRI) const {
  const LoadCommand *Command = getLoadCommandInfo(DRI.d.a);
  const SymtabLoadCommand *SymtabLoadCmd =
    reinterpret_cast<const SymtabLoadCommand*>(Command);
  return getSymbolTableEntryBase(DRI, SymtabLoadCmd);
}

const MachOObjectFileBase::SymbolTableEntryBase *
MachOObjectFileBase::getSymbolTableEntryBase(DataRefImpl DRI,
                                 const SymtabLoadCommand *SymtabLoadCmd) const {
  uint64_t SymbolTableOffset = SymtabLoadCmd->SymbolTableOffset;
  unsigned Index = DRI.d.b;

  unsigned SymbolTableEntrySize = is64Bit() ?
    sizeof(MachOObjectFile64Le::SymbolTableEntry) :
    sizeof(MachOObjectFile32Le::SymbolTableEntry);

  uint64_t Offset = SymbolTableOffset + Index * SymbolTableEntrySize;
  StringRef Data = getData(Offset, SymbolTableEntrySize);
  return reinterpret_cast<const SymbolTableEntryBase*>(Data.data());
}

error_code MachOObjectFileBase::getSymbolNext(DataRefImpl DRI,
                                              SymbolRef &Result) const {
  DRI.d.b++;
  moveToNextSymbol(DRI);
  Result = SymbolRef(DRI, this);
  return object_error::success;
}

error_code MachOObjectFileBase::getSymbolName(DataRefImpl DRI,
                                              StringRef &Result) const {
  const LoadCommand *Command = getLoadCommandInfo(DRI.d.a);
  const SymtabLoadCommand *SymtabLoadCmd =
    reinterpret_cast<const SymtabLoadCommand*>(Command);

  StringRef StringTable = getData(SymtabLoadCmd->StringTableOffset,
                                  SymtabLoadCmd->StringTableSize);

  const SymbolTableEntryBase *Entry =
    getSymbolTableEntryBase(DRI, SymtabLoadCmd);
  uint32_t StringIndex = Entry->StringIndex;

  const char *Start = &StringTable.data()[StringIndex];
  Result = StringRef(Start);

  return object_error::success;
}

error_code MachOObjectFileBase::getSymbolNMTypeChar(DataRefImpl DRI,
                                                    char &Result) const {
  const SymbolTableEntryBase *Entry = getSymbolTableEntryBase(DRI);
  uint8_t Type = Entry->Type;
  uint16_t Flags = Entry->Flags;

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
  Result = Char;
  return object_error::success;
}

error_code MachOObjectFileBase::getSymbolFlags(DataRefImpl DRI,
                                               uint32_t &Result) const {
  const SymbolTableEntryBase *Entry = getSymbolTableEntryBase(DRI);
  uint8_t MachOType = Entry->Type;
  uint16_t MachOFlags = Entry->Flags;

  // TODO: Correctly set SF_ThreadLocal
  Result = SymbolRef::SF_None;

  if ((MachOType & MachO::NlistMaskType) == MachO::NListTypeUndefined)
    Result |= SymbolRef::SF_Undefined;

  if (MachOFlags & macho::STF_StabsEntryMask)
    Result |= SymbolRef::SF_FormatSpecific;

  if (MachOType & MachO::NlistMaskExternal) {
    Result |= SymbolRef::SF_Global;
    if ((MachOType & MachO::NlistMaskType) == MachO::NListTypeUndefined)
      Result |= SymbolRef::SF_Common;
  }

  if (MachOFlags & (MachO::NListDescWeakRef | MachO::NListDescWeakDef))
    Result |= SymbolRef::SF_Weak;

  if ((MachOType & MachO::NlistMaskType) == MachO::NListTypeAbsolute)
    Result |= SymbolRef::SF_Absolute;

  return object_error::success;
}

error_code MachOObjectFileBase::getSymbolSection(DataRefImpl Symb,
                                                 section_iterator &Res) const {
  const SymbolTableEntryBase *Entry = getSymbolTableEntryBase(Symb);
  uint8_t index = Entry->SectionIndex;

  if (index == 0)
    Res = end_sections();
  else
    Res = section_iterator(SectionRef(Sections[index-1], this));

  return object_error::success;
}

error_code MachOObjectFileBase::getSymbolType(DataRefImpl Symb,
                                              SymbolRef::Type &Res) const {
  const SymbolTableEntryBase *Entry = getSymbolTableEntryBase(Symb);
  uint8_t n_type = Entry->Type;

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

error_code MachOObjectFileBase::getSymbolValue(DataRefImpl Symb,
                                               uint64_t &Val) const {
  report_fatal_error("getSymbolValue unimplemented in MachOObjectFileBase");
}

symbol_iterator MachOObjectFileBase::begin_symbols() const {
  // DRI.d.a = segment number; DRI.d.b = symbol index.
  DataRefImpl DRI;
  moveToNextSymbol(DRI);
  return symbol_iterator(SymbolRef(DRI, this));
}

symbol_iterator MachOObjectFileBase::end_symbols() const {
  DataRefImpl DRI;
  DRI.d.a = getHeader()->NumLoadCommands;
  return symbol_iterator(SymbolRef(DRI, this));
}

symbol_iterator MachOObjectFileBase::begin_dynamic_symbols() const {
  // TODO: implement
  report_fatal_error("Dynamic symbols unimplemented in MachOObjectFileBase");
}

symbol_iterator MachOObjectFileBase::end_dynamic_symbols() const {
  // TODO: implement
  report_fatal_error("Dynamic symbols unimplemented in MachOObjectFileBase");
}

library_iterator MachOObjectFileBase::begin_libraries_needed() const {
  // TODO: implement
  report_fatal_error("Needed libraries unimplemented in MachOObjectFileBase");
}

library_iterator MachOObjectFileBase::end_libraries_needed() const {
  // TODO: implement
  report_fatal_error("Needed libraries unimplemented in MachOObjectFileBase");
}

StringRef MachOObjectFileBase::getLoadName() const {
  // TODO: Implement
  report_fatal_error("get_load_name() unimplemented in MachOObjectFileBase");
}

/*===-- Sections ----------------------------------------------------------===*/

std::size_t MachOObjectFileBase::getSectionIndex(DataRefImpl Sec) const {
  SectionList::const_iterator loc =
    std::find(Sections.begin(), Sections.end(), Sec);
  assert(loc != Sections.end() && "Sec is not a valid section!");
  return std::distance(Sections.begin(), loc);
}

const MachOObjectFileBase::SectionBase*
MachOObjectFileBase::getSectionBase(DataRefImpl DRI) const {
  const LoadCommand *Command = getLoadCommandInfo(DRI.d.a);
  uintptr_t CommandAddr = reinterpret_cast<uintptr_t>(Command);

  bool Is64 = is64Bit();
  unsigned SegmentLoadSize =
    Is64 ? sizeof(MachOObjectFile64Le::SegmentLoadCommand) :
           sizeof(MachOObjectFile32Le::SegmentLoadCommand);
  unsigned SectionSize = Is64 ? sizeof(MachOObjectFile64Le::Section) :
                                sizeof(MachOObjectFile32Le::Section);

  uintptr_t SectionAddr = CommandAddr + SegmentLoadSize + DRI.d.b * SectionSize;
  return reinterpret_cast<const SectionBase*>(SectionAddr);
}

static StringRef parseSegmentOrSectionName(const char *P) {
  if (P[15] == 0)
    // Null terminated.
    return P;
  // Not null terminated, so this is a 16 char string.
  return StringRef(P, 16);
}

ArrayRef<char> MachOObjectFileBase::getSectionRawName(DataRefImpl DRI) const {
  const SectionBase *Base = getSectionBase(DRI);
  return ArrayRef<char>(Base->Name);
}

error_code MachOObjectFileBase::getSectionName(DataRefImpl DRI,
                                               StringRef &Result) const {
  ArrayRef<char> Raw = getSectionRawName(DRI);
  Result = parseSegmentOrSectionName(Raw.data());
  return object_error::success;
}

ArrayRef<char>
MachOObjectFileBase::getSectionRawFinalSegmentName(DataRefImpl Sec) const {
  const SectionBase *Base = getSectionBase(Sec);
  return ArrayRef<char>(Base->SegmentName);
}

StringRef
MachOObjectFileBase::getSectionFinalSegmentName(DataRefImpl DRI) const {
  ArrayRef<char> Raw = getSectionRawFinalSegmentName(DRI);
  return parseSegmentOrSectionName(Raw.data());
}

error_code MachOObjectFileBase::isSectionData(DataRefImpl DRI,
                                              bool &Result) const {
  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

error_code MachOObjectFileBase::isSectionBSS(DataRefImpl DRI,
                                             bool &Result) const {
  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

error_code
MachOObjectFileBase::isSectionRequiredForExecution(DataRefImpl Sec,
                                                   bool &Result) const {
  // FIXME: Unimplemented.
  Result = true;
  return object_error::success;
}

error_code MachOObjectFileBase::isSectionVirtual(DataRefImpl Sec,
                                                 bool &Result) const {
  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

error_code MachOObjectFileBase::isSectionReadOnlyData(DataRefImpl Sec,
                                                      bool &Result) const {
  // Consider using the code from isSectionText to look for __const sections.
  // Alternately, emit S_ATTR_PURE_INSTRUCTIONS and/or S_ATTR_SOME_INSTRUCTIONS
  // to use section attributes to distinguish code from data.

  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

relocation_iterator MachOObjectFileBase::getSectionRelBegin(DataRefImpl Sec) const {
  DataRefImpl ret;
  ret.d.b = getSectionIndex(Sec);
  return relocation_iterator(RelocationRef(ret, this));
}

section_iterator MachOObjectFileBase::end_sections() const {
  DataRefImpl DRI;
  DRI.d.a = getHeader()->NumLoadCommands;
  return section_iterator(SectionRef(DRI, this));
}

/*===-- Relocations -------------------------------------------------------===*/

error_code MachOObjectFileBase::getRelocationNext(DataRefImpl Rel,
                                                  RelocationRef &Res) const {
  ++Rel.d.a;
  Res = RelocationRef(Rel, this);
  return object_error::success;
}

// Helper to advance a section or symbol iterator multiple increments at a time.
template<class T>
error_code advance(T &it, size_t Val) {
  error_code ec;
  while (Val--) {
    it.increment(ec);
  }
  return ec;
}

template<class T>
void advanceTo(T &it, size_t Val) {
  if (error_code ec = advance(it, Val))
    report_fatal_error(ec.message());
}

void
MachOObjectFileBase::printRelocationTargetName(const RelocationEntry *RE,
                                               raw_string_ostream &fmt) const {
  // Target of a scattered relocation is an address.  In the interest of
  // generating pretty output, scan through the symbol table looking for a
  // symbol that aligns with that address.  If we find one, print it.
  // Otherwise, we just print the hex address of the target.
  if (isScattered(RE)) {
    uint32_t Val = RE->SymbolNum;

    error_code ec;
    for (symbol_iterator SI = begin_symbols(), SE = end_symbols(); SI != SE;
        SI.increment(ec)) {
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
    for (section_iterator SI = begin_sections(), SE = end_sections(); SI != SE;
         SI.increment(ec)) {
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
  bool isExtern = RE->External;
  uint32_t Val = RE->Address;

  if (isExtern) {
    symbol_iterator SI = begin_symbols();
    advanceTo(SI, Val);
    SI->getName(S);
  } else {
    section_iterator SI = begin_sections();
    advanceTo(SI, Val);
    SI->getName(S);
  }

  fmt << S;
}

error_code MachOObjectFileBase::getLibraryNext(DataRefImpl LibData,
                                               LibraryRef &Res) const {
  report_fatal_error("Needed libraries unimplemented in MachOObjectFileBase");
}

error_code MachOObjectFileBase::getLibraryPath(DataRefImpl LibData,
                                               StringRef &Res) const {
  report_fatal_error("Needed libraries unimplemented in MachOObjectFileBase");
}

error_code MachOObjectFileBase::getRelocationAdditionalInfo(DataRefImpl Rel,
                                                           int64_t &Res) const {
  Res = 0;
  return object_error::success;
}


/*===-- Miscellaneous -----------------------------------------------------===*/

uint8_t MachOObjectFileBase::getBytesInAddress() const {
  return is64Bit() ? 8 : 4;
}

StringRef MachOObjectFileBase::getFileFormatName() const {
  if (!is64Bit()) {
    switch (getHeader()->CPUType) {
    case llvm::MachO::CPUTypeI386:
      return "Mach-O 32-bit i386";
    case llvm::MachO::CPUTypeARM:
      return "Mach-O arm";
    case llvm::MachO::CPUTypePowerPC:
      return "Mach-O 32-bit ppc";
    default:
      assert((getHeader()->CPUType & llvm::MachO::CPUArchABI64) == 0 &&
             "64-bit object file when we're not 64-bit?");
      return "Mach-O 32-bit unknown";
    }
  }

  // Make sure the cpu type has the correct mask.
  assert((getHeader()->CPUType & llvm::MachO::CPUArchABI64)
	 == llvm::MachO::CPUArchABI64 &&
	 "32-bit object file when we're 64-bit?");

  switch (getHeader()->CPUType) {
  case llvm::MachO::CPUTypeX86_64:
    return "Mach-O 64-bit x86-64";
  case llvm::MachO::CPUTypePowerPC64:
    return "Mach-O 64-bit ppc64";
  default:
    return "Mach-O 64-bit unknown";
  }
}

unsigned MachOObjectFileBase::getArch() const {
  switch (getHeader()->CPUType) {
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
