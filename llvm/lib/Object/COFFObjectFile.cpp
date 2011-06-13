//===- COFFObjectFile.cpp - COFF object file implementation -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the COFFObjectFile class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/COFF.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"

using namespace llvm;
using namespace object;

namespace {
using support::ulittle8_t;
using support::ulittle16_t;
using support::ulittle32_t;
using support::little16_t;
}

SymbolRef COFFObjectFile::getSymbolNext(DataRefImpl Symb) const {
  const coff_symbol *symb = reinterpret_cast<const coff_symbol*>(Symb.p);
  symb += 1 + symb->NumberOfAuxSymbols;
  Symb.p = reinterpret_cast<intptr_t>(symb);
  return SymbolRef(Symb, this);
}

StringRef COFFObjectFile::getSymbolName(DataRefImpl Symb) const {
  const coff_symbol *symb = reinterpret_cast<const coff_symbol*>(Symb.p);
  // Check for string table entry. First 4 bytes are 0.
  if (symb->Name.Offset.Zeroes == 0) {
    uint32_t Offset = symb->Name.Offset.Offset;
    return StringRef(getString(Offset));
  }

  if (symb->Name.ShortName[7] == 0)
    // Null terminated, let ::strlen figure out the length.
    return StringRef(symb->Name.ShortName);
  // Not null terminated, use all 8 bytes.
  return StringRef(symb->Name.ShortName, 8);
}

uint64_t COFFObjectFile::getSymbolAddress(DataRefImpl Symb) const {
  const coff_symbol *symb = reinterpret_cast<const coff_symbol*>(Symb.p);
  const coff_section *Section = getSection(symb->SectionNumber);
  char Type = getSymbolNMTypeChar(Symb);
  if (Type == 'U' || Type == 'w')
    return UnknownAddressOrSize;
  if (Section)
    return Section->VirtualAddress + symb->Value;
  return symb->Value;
}

uint64_t COFFObjectFile::getSymbolSize(DataRefImpl Symb) const {
  // FIXME: Return the correct size. This requires looking at all the symbols
  //        in the same section as this symbol, and looking for either the next
  //        symbol, or the end of the section.
  const coff_symbol *symb = reinterpret_cast<const coff_symbol*>(Symb.p);
  const coff_section *Section = getSection(symb->SectionNumber);
  char Type = getSymbolNMTypeChar(Symb);
  if (Type == 'U' || Type == 'w')
    return UnknownAddressOrSize;
  if (Section)
    return Section->SizeOfRawData - symb->Value;
  return 0;
}

char COFFObjectFile::getSymbolNMTypeChar(DataRefImpl Symb) const {
  const coff_symbol *symb = reinterpret_cast<const coff_symbol*>(Symb.p);
  char ret = StringSwitch<char>(getSymbolName(Symb))
    .StartsWith(".debug", 'N')
    .StartsWith(".sxdata", 'N')
    .Default('?');

  if (ret != '?')
    return ret;

  uint32_t Characteristics = 0;
  if (const coff_section *Section = getSection(symb->SectionNumber)) {
    Characteristics = Section->Characteristics;
  }

  switch (symb->SectionNumber) {
  case COFF::IMAGE_SYM_UNDEFINED:
    // Check storage classes.
    if (symb->StorageClass == COFF::IMAGE_SYM_CLASS_WEAK_EXTERNAL)
      return 'w'; // Don't do ::toupper.
    else
      ret = 'u';
    break;
  case COFF::IMAGE_SYM_ABSOLUTE:
    ret = 'a';
    break;
  case COFF::IMAGE_SYM_DEBUG:
    ret = 'n';
    break;
  default:
    // Check section type.
    if (Characteristics & COFF::IMAGE_SCN_CNT_CODE)
      ret = 't';
    else if (  Characteristics & COFF::IMAGE_SCN_MEM_READ
            && ~Characteristics & COFF::IMAGE_SCN_MEM_WRITE) // Read only.
      ret = 'r';
    else if (Characteristics & COFF::IMAGE_SCN_CNT_INITIALIZED_DATA)
      ret = 'd';
    else if (Characteristics & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA)
      ret = 'b';
    else if (Characteristics & COFF::IMAGE_SCN_LNK_INFO)
      ret = 'i';

    // Check for section symbol.
    else if (  symb->StorageClass == COFF::IMAGE_SYM_CLASS_STATIC
            && symb->Value == 0)
       ret = 's';
  }

  if (symb->StorageClass == COFF::IMAGE_SYM_CLASS_EXTERNAL)
    ret = ::toupper(ret);

  return ret;
}

bool COFFObjectFile::isSymbolInternal(DataRefImpl Symb) const {
  return false;
}

SectionRef COFFObjectFile::getSectionNext(DataRefImpl Sec) const {
  const coff_section *sec = reinterpret_cast<const coff_section*>(Sec.p);
  sec += 1;
  Sec.p = reinterpret_cast<intptr_t>(sec);
  return SectionRef(Sec, this);
}

StringRef COFFObjectFile::getSectionName(DataRefImpl Sec) const {
  const coff_section *sec = reinterpret_cast<const coff_section*>(Sec.p);
  StringRef name;
  if (sec->Name[7] == 0)
    // Null terminated, let ::strlen figure out the length.
    name = sec->Name;
  else
    // Not null terminated, use all 8 bytes.
    name = StringRef(sec->Name, 8);

  // Check for string table entry. First byte is '/'.
  if (name[0] == '/') {
    uint32_t Offset;
    name.substr(1).getAsInteger(10, Offset);
    return StringRef(getString(Offset));
  }

  // It's just a normal name.
  return name;
}

uint64_t COFFObjectFile::getSectionAddress(DataRefImpl Sec) const {
  const coff_section *sec = reinterpret_cast<const coff_section*>(Sec.p);
  return sec->VirtualAddress;
}

uint64_t COFFObjectFile::getSectionSize(DataRefImpl Sec) const {
  const coff_section *sec = reinterpret_cast<const coff_section*>(Sec.p);
  return sec->SizeOfRawData;
}

StringRef COFFObjectFile::getSectionContents(DataRefImpl Sec) const {
  const coff_section *sec = reinterpret_cast<const coff_section*>(Sec.p);
  return StringRef(reinterpret_cast<const char *>(base()
                   + sec->PointerToRawData),
                   sec->SizeOfRawData);
}

bool COFFObjectFile::isSectionText(DataRefImpl Sec) const {
  const coff_section *sec = reinterpret_cast<const coff_section*>(Sec.p);
  return sec->Characteristics & COFF::IMAGE_SCN_CNT_CODE;
}

COFFObjectFile::COFFObjectFile(MemoryBuffer *Object, error_code &ec)
  : ObjectFile(Binary::isCOFF, Object, ec) {

  HeaderOff = 0;

  if (base()[0] == 0x4d && base()[1] == 0x5a) {
    // PE/COFF, seek through MS-DOS compatibility stub and 4-byte
    // PE signature to find 'normal' COFF header.
    HeaderOff += *reinterpret_cast<const ulittle32_t *>(base() + 0x3c);
    HeaderOff += 4;
  }

  Header = reinterpret_cast<const coff_file_header *>(base() + HeaderOff);
  SectionTable =
    reinterpret_cast<const coff_section *>( base()
                                          + HeaderOff
                                          + sizeof(coff_file_header)
                                          + Header->SizeOfOptionalHeader);
  SymbolTable =
    reinterpret_cast<const coff_symbol *>(base()
    + Header->PointerToSymbolTable);

  // Find string table.
  StringTable = reinterpret_cast<const char *>(base())
              + Header->PointerToSymbolTable
              + Header->NumberOfSymbols * 18;
}

ObjectFile::symbol_iterator COFFObjectFile::begin_symbols() const {
  DataRefImpl ret;
  memset(&ret, 0, sizeof(DataRefImpl));
  ret.p = reinterpret_cast<intptr_t>(SymbolTable);
  return symbol_iterator(SymbolRef(ret, this));
}

ObjectFile::symbol_iterator COFFObjectFile::end_symbols() const {
  // The symbol table ends where the string table begins.
  DataRefImpl ret;
  memset(&ret, 0, sizeof(DataRefImpl));
  ret.p = reinterpret_cast<intptr_t>(StringTable);
  return symbol_iterator(SymbolRef(ret, this));
}

ObjectFile::section_iterator COFFObjectFile::begin_sections() const {
  DataRefImpl ret;
  memset(&ret, 0, sizeof(DataRefImpl));
  ret.p = reinterpret_cast<intptr_t>(SectionTable);
  return section_iterator(SectionRef(ret, this));
}

ObjectFile::section_iterator COFFObjectFile::end_sections() const {
  DataRefImpl ret;
  memset(&ret, 0, sizeof(DataRefImpl));
  ret.p = reinterpret_cast<intptr_t>(SectionTable + Header->NumberOfSections);
  return section_iterator(SectionRef(ret, this));
}

uint8_t COFFObjectFile::getBytesInAddress() const {
  return getArch() == Triple::x86_64 ? 8 : 4;
}

StringRef COFFObjectFile::getFileFormatName() const {
  switch(Header->Machine) {
  case COFF::IMAGE_FILE_MACHINE_I386:
    return "COFF-i386";
  case COFF::IMAGE_FILE_MACHINE_AMD64:
    return "COFF-x86-64";
  default:
    return "COFF-<unknown arch>";
  }
}

unsigned COFFObjectFile::getArch() const {
  switch(Header->Machine) {
  case COFF::IMAGE_FILE_MACHINE_I386:
    return Triple::x86;
  case COFF::IMAGE_FILE_MACHINE_AMD64:
    return Triple::x86_64;
  default:
    return Triple::UnknownArch;
  }
}

const coff_section *COFFObjectFile::getSection(std::size_t index) const {
  if (index > 0 && index <= Header->NumberOfSections)
    return SectionTable + (index - 1);
  return 0;
}

const char *COFFObjectFile::getString(std::size_t offset) const {
  const ulittle32_t *StringTableSize =
    reinterpret_cast<const ulittle32_t *>(StringTable);
  if (offset < *StringTableSize)
    return StringTable + offset;
  return 0;
}

namespace llvm {

  ObjectFile *ObjectFile::createCOFFObjectFile(MemoryBuffer *Object) {
    error_code ec;
    return new COFFObjectFile(Object, ec);
  }

} // end namespace llvm
