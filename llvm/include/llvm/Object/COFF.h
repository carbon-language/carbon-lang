//===- COFF.h - COFF object file implementation -----------------*- C++ -*-===//
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

#ifndef LLVM_OBJECT_COFF_H
#define LLVM_OBJECT_COFF_H

#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Endian.h"

namespace llvm {
namespace object {

struct coff_file_header {
  support::ulittle16_t Machine;
  support::ulittle16_t NumberOfSections;
  support::ulittle32_t TimeDateStamp;
  support::ulittle32_t PointerToSymbolTable;
  support::ulittle32_t NumberOfSymbols;
  support::ulittle16_t SizeOfOptionalHeader;
  support::ulittle16_t Characteristics;
};

struct coff_symbol {
  struct StringTableOffset {
    support::ulittle32_t Zeroes;
    support::ulittle32_t Offset;
  };

  union {
    char ShortName[8];
    StringTableOffset Offset;
  } Name;

  support::ulittle32_t Value;
  support::little16_t SectionNumber;

  struct {
    support::ulittle8_t BaseType;
    support::ulittle8_t ComplexType;
  } Type;

  support::ulittle8_t  StorageClass;
  support::ulittle8_t  NumberOfAuxSymbols;
};

struct coff_section {
  char Name[8];
  support::ulittle32_t VirtualSize;
  support::ulittle32_t VirtualAddress;
  support::ulittle32_t SizeOfRawData;
  support::ulittle32_t PointerToRawData;
  support::ulittle32_t PointerToRelocations;
  support::ulittle32_t PointerToLinenumbers;
  support::ulittle16_t NumberOfRelocations;
  support::ulittle16_t NumberOfLinenumbers;
  support::ulittle32_t Characteristics;
};

class COFFObjectFile : public ObjectFile {
private:
        uint64_t         HeaderOff;
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
  COFFObjectFile(MemoryBuffer *Object, error_code &ec);
  virtual symbol_iterator begin_symbols() const;
  virtual symbol_iterator end_symbols() const;
  virtual section_iterator begin_sections() const;
  virtual section_iterator end_sections() const;

  virtual uint8_t getBytesInAddress() const;
  virtual StringRef getFileFormatName() const;
  virtual unsigned getArch() const;
};

}
}

#endif
