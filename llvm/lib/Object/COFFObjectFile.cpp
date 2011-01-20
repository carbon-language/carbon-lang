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

#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace object;

namespace {
using support::ulittle8_t;
using support::ulittle16_t;
using support::ulittle32_t;
using support::little16_t;
}

namespace {
struct coff_file_header {
  ulittle16_t Machine;
  ulittle16_t NumberOfSections;
  ulittle32_t TimeDateStamp;
  ulittle32_t PointerToSymbolTable;
  ulittle32_t NumberOfSymbols;
  ulittle16_t SizeOfOptionalHeader;
  ulittle16_t Characteristics;
};
}

extern char coff_file_header_layout_static_assert
            [sizeof(coff_file_header) == 20 ? 1 : -1];

namespace {
struct coff_symbol {
  struct StringTableOffset {
    ulittle32_t Zeroes;
    ulittle32_t Offset;
  };

  union {
    char ShortName[8];
    StringTableOffset Offset;
  } Name;

  ulittle32_t Value;
  little16_t SectionNumber;

  struct {
    ulittle8_t BaseType;
    ulittle8_t ComplexType;
  } Type;

  ulittle8_t  StorageClass;
  ulittle8_t  NumberOfAuxSymbols;
};
}

extern char coff_coff_symbol_layout_static_assert
            [sizeof(coff_symbol) == 18 ? 1 : -1];

namespace {
struct coff_section {
  char Name[8];
  ulittle32_t VirtualSize;
  ulittle32_t VirtualAddress;
  ulittle32_t SizeOfRawData;
  ulittle32_t PointerToRawData;
  ulittle32_t PointerToRelocations;
  ulittle32_t PointerToLinenumbers;
  ulittle16_t NumberOfRelocations;
  ulittle16_t NumberOfLinenumbers;
  ulittle32_t Characteristics;
};
}

extern char coff_coff_section_layout_static_assert
            [sizeof(coff_section) == 40 ? 1 : -1];

namespace {
class COFFObjectFile : public ObjectFile {
private:
  const coff_file_header *Header;
  const coff_section     *SectionTable;
  const coff_symbol      *SymbolTable;
  const char             *StringTable;

  const coff_section     *getSection(std::size_t index) const;
  const char             *getString(std::size_t offset) const;

protected:
  virtual SymbolRef getSymbolNext(DataRefImpl Symb) const;
  virtual StringRef getSymbolName(DataRefImpl Symb) const;
  virtual uint64_t  getSymbolAddress(DataRefImpl Symb) const;
  virtual uint64_t  getSymbolSize(DataRefImpl Symb) const;
  virtual char      getSymbolNMTypeChar(DataRefImpl Symb) const;
  virtual bool      isSymbolInternal(DataRefImpl Symb) const;

  virtual SectionRef getSectionNext(DataRefImpl Sec) const;
  virtual StringRef  getSectionName(DataRefImpl Sec) const;
  virtual uint64_t   getSectionAddress(DataRefImpl Sec) const;
  virtual uint64_t   getSectionSize(DataRefImpl Sec) const;
  virtual StringRef  getSectionContents(DataRefImpl Sec) const;
  virtual bool       isSectionText(DataRefImpl Sec) const;

public:
  COFFObjectFile(MemoryBuffer *Object);
  virtual symbol_iterator begin_symbols() const;
  virtual symbol_iterator end_symbols() const;
  virtual section_iterator begin_sections() const;
  virtual section_iterator end_sections() const;

  virtual uint8_t getBytesInAddress() const;
  virtual StringRef getFileFormatName() const;
  virtual unsigned getArch() const;
};
} // end namespace

SymbolRef COFFObjectFile::getSymbolNext(DataRefImpl Symb) const {
  const coff_symbol *symb = *reinterpret_cast<const coff_symbol**>(&Symb);
  symb += 1 + symb->NumberOfAuxSymbols;
  return SymbolRef(DataRefImpl(symb), this);
}

StringRef COFFObjectFile::getSymbolName(DataRefImpl Symb) const {
  const coff_symbol *symb = *reinterpret_cast<const coff_symbol**>(&Symb);
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
  const coff_symbol *symb = *reinterpret_cast<const coff_symbol**>(&Symb);
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
  const coff_symbol *symb = *reinterpret_cast<const coff_symbol**>(&Symb);
  const coff_section *Section = getSection(symb->SectionNumber);
  char Type = getSymbolNMTypeChar(Symb);
  if (Type == 'U' || Type == 'w')
    return UnknownAddressOrSize;
  if (Section)
    return Section->SizeOfRawData - symb->Value;
  return 0;
}

char COFFObjectFile::getSymbolNMTypeChar(DataRefImpl Symb) const {
  const coff_symbol *symb = *reinterpret_cast<const coff_symbol**>(&Symb);
  char ret = StringSwitch<char>(getSymbolName(Symb))
    .StartsWith(".debug", 'N')
    .StartsWith(".sxdata", 'N')
    .Default('?');

  if (ret != '?')
    return ret;

  uint32_t Characteristics = 0;
  uint32_t PointerToRawData = 0;
  const coff_section *Section = getSection(symb->SectionNumber);
  if (Section) {
    Characteristics = Section->Characteristics;
    PointerToRawData = Section->PointerToRawData;
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
  const coff_section *sec = *reinterpret_cast<const coff_section**>(&Sec);
  sec += 1;
  return SectionRef(DataRefImpl(sec), this);
}

StringRef COFFObjectFile::getSectionName(DataRefImpl Sec) const {
  const coff_section *sec = *reinterpret_cast<const coff_section**>(&Sec);
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
    name.getAsInteger(10, Offset);
    return StringRef(getString(Offset));
  }

  // It's just a normal name.
  return name;
}

uint64_t COFFObjectFile::getSectionAddress(DataRefImpl Sec) const {
  const coff_section *sec = *reinterpret_cast<const coff_section**>(&Sec);
  return sec->VirtualAddress;
}

uint64_t COFFObjectFile::getSectionSize(DataRefImpl Sec) const {
  const coff_section *sec = *reinterpret_cast<const coff_section**>(&Sec);
  return sec->SizeOfRawData;
}

StringRef COFFObjectFile::getSectionContents(DataRefImpl Sec) const {
  const coff_section *sec = *reinterpret_cast<const coff_section**>(&Sec);
  return StringRef(reinterpret_cast<const char *>(base + sec->PointerToRawData),
                   sec->SizeOfRawData);
}

bool COFFObjectFile::isSectionText(DataRefImpl Sec) const {
  const coff_section *sec = *reinterpret_cast<const coff_section**>(&Sec);
  return sec->Characteristics & COFF::IMAGE_SCN_CNT_CODE;
}

COFFObjectFile::COFFObjectFile(MemoryBuffer *Object)
  : ObjectFile(Object) {
  Header = reinterpret_cast<const coff_file_header *>(base);
  SectionTable =
    reinterpret_cast<const coff_section *>( base
                                          + sizeof(coff_file_header)
                                          + Header->SizeOfOptionalHeader);
  SymbolTable =
    reinterpret_cast<const coff_symbol *>(base + Header->PointerToSymbolTable);

  // Find string table.
  StringTable = reinterpret_cast<const char *>(base)
              + Header->PointerToSymbolTable
              + Header->NumberOfSymbols * 18;
}

ObjectFile::symbol_iterator COFFObjectFile::begin_symbols() const {
  return symbol_iterator(
    SymbolRef(DataRefImpl(SymbolTable), this));
}

ObjectFile::symbol_iterator COFFObjectFile::end_symbols() const {
  // The symbol table ends where the string table begins.
  return symbol_iterator(
    SymbolRef(DataRefImpl(StringTable), this));
}

ObjectFile::section_iterator COFFObjectFile::begin_sections() const {
  return section_iterator(
    SectionRef(DataRefImpl(SectionTable), this));
}

ObjectFile::section_iterator COFFObjectFile::end_sections() const {
  return section_iterator(
    SectionRef(
      DataRefImpl((void *)(SectionTable + Header->NumberOfSections)), this));
}

uint8_t COFFObjectFile::getBytesInAddress() const {
  return 4;
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
    return new COFFObjectFile(Object);
  }

} // end namespace llvm
