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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>

using namespace llvm;
using namespace object;

namespace {
using support::ulittle8_t;
using support::ulittle16_t;
using support::ulittle32_t;
using support::little16_t;
}

namespace {
// Returns false if size is greater than the buffer size. And sets ec.
bool checkSize(const MemoryBuffer *m, error_code &ec, uint64_t size) {
  if (m->getBufferSize() < size) {
    ec = object_error::unexpected_eof;
    return false;
  }
  return true;
}

// Sets Obj unless any bytes in [addr, addr + size) fall outsize of m.
// Returns unexpected_eof if error.
template<typename T>
error_code getObject(const T *&Obj, const MemoryBuffer *M, const uint8_t *Ptr,
                     const size_t Size = sizeof(T)) {
  uintptr_t Addr = uintptr_t(Ptr);
  if (Addr + Size < Addr ||
      Addr + Size < Size ||
      Addr + Size > uintptr_t(M->getBufferEnd())) {
    return object_error::unexpected_eof;
  }
  Obj = reinterpret_cast<const T *>(Addr);
  return object_error::success;
}
}

const coff_symbol *COFFObjectFile::toSymb(DataRefImpl Symb) const {
  const coff_symbol *addr = reinterpret_cast<const coff_symbol*>(Symb.p);

# ifndef NDEBUG
  // Verify that the symbol points to a valid entry in the symbol table.
  uintptr_t offset = uintptr_t(addr) - uintptr_t(base());
  if (offset < COFFHeader->PointerToSymbolTable
      || offset >= COFFHeader->PointerToSymbolTable
         + (COFFHeader->NumberOfSymbols * sizeof(coff_symbol)))
    report_fatal_error("Symbol was outside of symbol table.");

  assert((offset - COFFHeader->PointerToSymbolTable) % sizeof(coff_symbol)
         == 0 && "Symbol did not point to the beginning of a symbol");
# endif

  return addr;
}

const coff_section *COFFObjectFile::toSec(DataRefImpl Sec) const {
  const coff_section *addr = reinterpret_cast<const coff_section*>(Sec.p);

# ifndef NDEBUG
  // Verify that the section points to a valid entry in the section table.
  if (addr < SectionTable
      || addr >= (SectionTable + COFFHeader->NumberOfSections))
    report_fatal_error("Section was outside of section table.");

  uintptr_t offset = uintptr_t(addr) - uintptr_t(SectionTable);
  assert(offset % sizeof(coff_section) == 0 &&
         "Section did not point to the beginning of a section");
# endif

  return addr;
}

error_code COFFObjectFile::getSymbolNext(DataRefImpl Symb,
                                         SymbolRef &Result) const {
  const coff_symbol *symb = toSymb(Symb);
  symb += 1 + symb->NumberOfAuxSymbols;
  Symb.p = reinterpret_cast<uintptr_t>(symb);
  Result = SymbolRef(Symb, this);
  return object_error::success;
}

 error_code COFFObjectFile::getSymbolName(DataRefImpl Symb,
                                          StringRef &Result) const {
  const coff_symbol *symb = toSymb(Symb);
  return getSymbolName(symb, Result);
}

error_code COFFObjectFile::getSymbolFileOffset(DataRefImpl Symb,
                                            uint64_t &Result) const {
  const coff_symbol *symb = toSymb(Symb);
  const coff_section *Section = NULL;
  if (error_code ec = getSection(symb->SectionNumber, Section))
    return ec;

  if (symb->SectionNumber == COFF::IMAGE_SYM_UNDEFINED)
    Result = UnknownAddressOrSize;
  else if (Section)
    Result = Section->PointerToRawData + symb->Value;
  else
    Result = symb->Value;
  return object_error::success;
}

error_code COFFObjectFile::getSymbolAddress(DataRefImpl Symb,
                                            uint64_t &Result) const {
  const coff_symbol *symb = toSymb(Symb);
  const coff_section *Section = NULL;
  if (error_code ec = getSection(symb->SectionNumber, Section))
    return ec;

  if (symb->SectionNumber == COFF::IMAGE_SYM_UNDEFINED)
    Result = UnknownAddressOrSize;
  else if (Section)
    Result = Section->VirtualAddress + symb->Value;
  else
    Result = symb->Value;
  return object_error::success;
}

error_code COFFObjectFile::getSymbolType(DataRefImpl Symb,
                                         SymbolRef::Type &Result) const {
  const coff_symbol *symb = toSymb(Symb);
  Result = SymbolRef::ST_Other;
  if (symb->StorageClass == COFF::IMAGE_SYM_CLASS_EXTERNAL &&
      symb->SectionNumber == COFF::IMAGE_SYM_UNDEFINED) {
    Result = SymbolRef::ST_Unknown;
  } else {
    if (symb->getComplexType() == COFF::IMAGE_SYM_DTYPE_FUNCTION) {
      Result = SymbolRef::ST_Function;
    } else {
      uint32_t Characteristics = 0;
      if (symb->SectionNumber > 0) {
        const coff_section *Section = NULL;
        if (error_code ec = getSection(symb->SectionNumber, Section))
          return ec;
        Characteristics = Section->Characteristics;
      }
      if (Characteristics & COFF::IMAGE_SCN_MEM_READ &&
          ~Characteristics & COFF::IMAGE_SCN_MEM_WRITE) // Read only.
        Result = SymbolRef::ST_Data;
    }
  }
  return object_error::success;
}

error_code COFFObjectFile::getSymbolFlags(DataRefImpl Symb,
                                          uint32_t &Result) const {
  const coff_symbol *symb = toSymb(Symb);
  Result = SymbolRef::SF_None;

  // TODO: Correctly set SF_FormatSpecific, SF_ThreadLocal, SF_Common

  if (symb->StorageClass == COFF::IMAGE_SYM_CLASS_EXTERNAL &&
      symb->SectionNumber == COFF::IMAGE_SYM_UNDEFINED)
    Result |= SymbolRef::SF_Undefined;

  // TODO: This are certainly too restrictive.
  if (symb->StorageClass == COFF::IMAGE_SYM_CLASS_EXTERNAL)
    Result |= SymbolRef::SF_Global;

  if (symb->StorageClass == COFF::IMAGE_SYM_CLASS_WEAK_EXTERNAL)
    Result |= SymbolRef::SF_Weak;

  if (symb->SectionNumber == COFF::IMAGE_SYM_ABSOLUTE)
    Result |= SymbolRef::SF_Absolute;

  return object_error::success;
}

error_code COFFObjectFile::getSymbolSize(DataRefImpl Symb,
                                         uint64_t &Result) const {
  // FIXME: Return the correct size. This requires looking at all the symbols
  //        in the same section as this symbol, and looking for either the next
  //        symbol, or the end of the section.
  const coff_symbol *symb = toSymb(Symb);
  const coff_section *Section = NULL;
  if (error_code ec = getSection(symb->SectionNumber, Section))
    return ec;

  if (symb->SectionNumber == COFF::IMAGE_SYM_UNDEFINED)
    Result = UnknownAddressOrSize;
  else if (Section)
    Result = Section->SizeOfRawData - symb->Value;
  else
    Result = 0;
  return object_error::success;
}

error_code COFFObjectFile::getSymbolSection(DataRefImpl Symb,
                                            section_iterator &Result) const {
  const coff_symbol *symb = toSymb(Symb);
  if (symb->SectionNumber <= COFF::IMAGE_SYM_UNDEFINED)
    Result = end_sections();
  else {
    const coff_section *sec = 0;
    if (error_code ec = getSection(symb->SectionNumber, sec)) return ec;
    DataRefImpl Sec;
    Sec.p = reinterpret_cast<uintptr_t>(sec);
    Result = section_iterator(SectionRef(Sec, this));
  }
  return object_error::success;
}

error_code COFFObjectFile::getSymbolValue(DataRefImpl Symb,
                                          uint64_t &Val) const {
  report_fatal_error("getSymbolValue unimplemented in COFFObjectFile");
}

error_code COFFObjectFile::getSectionNext(DataRefImpl Sec,
                                          SectionRef &Result) const {
  const coff_section *sec = toSec(Sec);
  sec += 1;
  Sec.p = reinterpret_cast<uintptr_t>(sec);
  Result = SectionRef(Sec, this);
  return object_error::success;
}

error_code COFFObjectFile::getSectionName(DataRefImpl Sec,
                                          StringRef &Result) const {
  const coff_section *sec = toSec(Sec);
  return getSectionName(sec, Result);
}

error_code COFFObjectFile::getSectionAddress(DataRefImpl Sec,
                                             uint64_t &Result) const {
  const coff_section *sec = toSec(Sec);
  Result = sec->VirtualAddress;
  return object_error::success;
}

error_code COFFObjectFile::getSectionSize(DataRefImpl Sec,
                                          uint64_t &Result) const {
  const coff_section *sec = toSec(Sec);
  Result = sec->SizeOfRawData;
  return object_error::success;
}

error_code COFFObjectFile::getSectionContents(DataRefImpl Sec,
                                              StringRef &Result) const {
  const coff_section *sec = toSec(Sec);
  ArrayRef<uint8_t> Res;
  error_code EC = getSectionContents(sec, Res);
  Result = StringRef(reinterpret_cast<const char*>(Res.data()), Res.size());
  return EC;
}

error_code COFFObjectFile::getSectionAlignment(DataRefImpl Sec,
                                               uint64_t &Res) const {
  const coff_section *sec = toSec(Sec);
  if (!sec)
    return object_error::parse_failed;
  Res = uint64_t(1) << (((sec->Characteristics & 0x00F00000) >> 20) - 1);
  return object_error::success;
}

error_code COFFObjectFile::isSectionText(DataRefImpl Sec,
                                         bool &Result) const {
  const coff_section *sec = toSec(Sec);
  Result = sec->Characteristics & COFF::IMAGE_SCN_CNT_CODE;
  return object_error::success;
}

error_code COFFObjectFile::isSectionData(DataRefImpl Sec,
                                         bool &Result) const {
  const coff_section *sec = toSec(Sec);
  Result = sec->Characteristics & COFF::IMAGE_SCN_CNT_INITIALIZED_DATA;
  return object_error::success;
}

error_code COFFObjectFile::isSectionBSS(DataRefImpl Sec,
                                        bool &Result) const {
  const coff_section *sec = toSec(Sec);
  Result = sec->Characteristics & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA;
  return object_error::success;
}

error_code COFFObjectFile::isSectionRequiredForExecution(DataRefImpl Sec,
                                                         bool &Result) const {
  // FIXME: Unimplemented
  Result = true;
  return object_error::success;
}

error_code COFFObjectFile::isSectionVirtual(DataRefImpl Sec,
                                           bool &Result) const {
  const coff_section *sec = toSec(Sec);
  Result = sec->Characteristics & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA;
  return object_error::success;
}

error_code COFFObjectFile::isSectionZeroInit(DataRefImpl Sec,
                                             bool &Result) const {
  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

error_code COFFObjectFile::isSectionReadOnlyData(DataRefImpl Sec,
                                                bool &Result) const {
  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

error_code COFFObjectFile::sectionContainsSymbol(DataRefImpl Sec,
                                                 DataRefImpl Symb,
                                                 bool &Result) const {
  const coff_section *sec = toSec(Sec);
  const coff_symbol *symb = toSymb(Symb);
  const coff_section *symb_sec = 0;
  if (error_code ec = getSection(symb->SectionNumber, symb_sec)) return ec;
  if (symb_sec == sec)
    Result = true;
  else
    Result = false;
  return object_error::success;
}

relocation_iterator COFFObjectFile::section_rel_begin(DataRefImpl Sec) const {
  const coff_section *sec = toSec(Sec);
  DataRefImpl ret;
  if (sec->NumberOfRelocations == 0)
    ret.p = 0;
  else
    ret.p = reinterpret_cast<uintptr_t>(base() + sec->PointerToRelocations);

  return relocation_iterator(RelocationRef(ret, this));
}

relocation_iterator COFFObjectFile::section_rel_end(DataRefImpl Sec) const {
  const coff_section *sec = toSec(Sec);
  DataRefImpl ret;
  if (sec->NumberOfRelocations == 0)
    ret.p = 0;
  else
    ret.p = reinterpret_cast<uintptr_t>(
              reinterpret_cast<const coff_relocation*>(
                base() + sec->PointerToRelocations)
              + sec->NumberOfRelocations);

  return relocation_iterator(RelocationRef(ret, this));
}

// Initialize the pointer to the symbol table.
error_code COFFObjectFile::initSymbolTablePtr() {
  if (error_code ec = getObject(
          SymbolTable, Data, base() + COFFHeader->PointerToSymbolTable,
          COFFHeader->NumberOfSymbols * sizeof(coff_symbol)))
    return ec;

  // Find string table. The first four byte of the string table contains the
  // total size of the string table, including the size field itself. If the
  // string table is empty, the value of the first four byte would be 4.
  const uint8_t *StringTableAddr =
      base() + COFFHeader->PointerToSymbolTable +
      COFFHeader->NumberOfSymbols * sizeof(coff_symbol);
  const ulittle32_t *StringTableSizePtr;
  if (error_code ec = getObject(StringTableSizePtr, Data, StringTableAddr))
    return ec;
  StringTableSize = *StringTableSizePtr;
  if (error_code ec =
      getObject(StringTable, Data, StringTableAddr, StringTableSize))
    return ec;

  // Check that the string table is null terminated if has any in it.
  if (StringTableSize < 4 ||
      (StringTableSize > 4 && StringTable[StringTableSize - 1] != 0))
    return  object_error::parse_failed;
  return object_error::success;
}

// Returns the file offset for the given RVA.
error_code COFFObjectFile::getRvaPtr(uint32_t Rva, uintptr_t &Res) const {
  error_code ec;
  for (section_iterator i = begin_sections(), e = end_sections(); i != e;
       i.increment(ec)) {
    if (ec)
      return ec;
    const coff_section *Section = getCOFFSection(i);
    uint32_t SectionStart = Section->VirtualAddress;
    uint32_t SectionEnd = Section->VirtualAddress + Section->VirtualSize;
    if (SectionStart <= Rva && Rva < SectionEnd) {
      uint32_t Offset = Rva - SectionStart;
      Res = uintptr_t(base()) + Section->PointerToRawData + Offset;
      return object_error::success;
    }
  }
  return object_error::parse_failed;
}

// Returns hint and name fields, assuming \p Rva is pointing to a Hint/Name
// table entry.
error_code COFFObjectFile::
getHintName(uint32_t Rva, uint16_t &Hint, StringRef &Name) const {
  uintptr_t IntPtr = 0;
  if (error_code ec = getRvaPtr(Rva, IntPtr))
    return ec;
  const uint8_t *Ptr = reinterpret_cast<const uint8_t *>(IntPtr);
  Hint = *reinterpret_cast<const ulittle16_t *>(Ptr);
  Name = StringRef(reinterpret_cast<const char *>(Ptr + 2));
  return object_error::success;
}

// Find the import table.
error_code COFFObjectFile::initImportTablePtr() {
  // First, we get the RVA of the import table. If the file lacks a pointer to
  // the import table, do nothing.
  const data_directory *DataEntry;
  if (getDataDirectory(COFF::IMPORT_TABLE, DataEntry))
    return object_error::success;

  // Do nothing if the pointer to import table is NULL.
  if (DataEntry->RelativeVirtualAddress == 0)
    return object_error::success;

  uint32_t ImportTableRva = DataEntry->RelativeVirtualAddress;
  NumberOfImportDirectory = DataEntry->Size /
      sizeof(import_directory_table_entry);

  // Find the section that contains the RVA. This is needed because the RVA is
  // the import table's memory address which is different from its file offset.
  uintptr_t IntPtr = 0;
  if (error_code ec = getRvaPtr(ImportTableRva, IntPtr))
    return ec;
  ImportDirectory = reinterpret_cast<
      const import_directory_table_entry *>(IntPtr);
  return object_error::success;
}

// Find the export table.
error_code COFFObjectFile::initExportTablePtr() {
  // First, we get the RVA of the export table. If the file lacks a pointer to
  // the export table, do nothing.
  const data_directory *DataEntry;
  if (getDataDirectory(COFF::EXPORT_TABLE, DataEntry))
    return object_error::success;

  // Do nothing if the pointer to export table is NULL.
  if (DataEntry->RelativeVirtualAddress == 0)
    return object_error::success;

  uint32_t ExportTableRva = DataEntry->RelativeVirtualAddress;
  uintptr_t IntPtr = 0;
  if (error_code EC = getRvaPtr(ExportTableRva, IntPtr))
    return EC;
  ExportDirectory = reinterpret_cast<const export_directory_table_entry *>(IntPtr);
  return object_error::success;
}

COFFObjectFile::COFFObjectFile(MemoryBuffer *Object, error_code &ec)
  : ObjectFile(Binary::ID_COFF, Object)
  , COFFHeader(0)
  , PE32Header(0)
  , DataDirectory(0)
  , SectionTable(0)
  , SymbolTable(0)
  , StringTable(0)
  , StringTableSize(0)
  , ImportDirectory(0)
  , NumberOfImportDirectory(0)
  , ExportDirectory(0) {
  // Check that we at least have enough room for a header.
  if (!checkSize(Data, ec, sizeof(coff_file_header))) return;

  // The current location in the file where we are looking at.
  uint64_t CurPtr = 0;

  // PE header is optional and is present only in executables. If it exists,
  // it is placed right after COFF header.
  bool hasPEHeader = false;

  // Check if this is a PE/COFF file.
  if (base()[0] == 0x4d && base()[1] == 0x5a) {
    // PE/COFF, seek through MS-DOS compatibility stub and 4-byte
    // PE signature to find 'normal' COFF header.
    if (!checkSize(Data, ec, 0x3c + 8)) return;
    CurPtr = *reinterpret_cast<const ulittle16_t *>(base() + 0x3c);
    // Check the PE magic bytes. ("PE\0\0")
    if (std::memcmp(base() + CurPtr, "PE\0\0", 4) != 0) {
      ec = object_error::parse_failed;
      return;
    }
    CurPtr += 4; // Skip the PE magic bytes.
    hasPEHeader = true;
  }

  if ((ec = getObject(COFFHeader, Data, base() + CurPtr)))
    return;
  CurPtr += sizeof(coff_file_header);

  if (hasPEHeader) {
    if ((ec = getObject(PE32Header, Data, base() + CurPtr)))
      return;
    if (PE32Header->Magic != 0x10b) {
      // We only support PE32. If this is PE32 (not PE32+), the magic byte
      // should be 0x10b. If this is not PE32, continue as if there's no PE
      // header in this file.
      PE32Header = 0;
    } else if (PE32Header->NumberOfRvaAndSize > 0) {
      const uint8_t *addr = base() + CurPtr + sizeof(pe32_header);
      uint64_t size = sizeof(data_directory) * PE32Header->NumberOfRvaAndSize;
      if ((ec = getObject(DataDirectory, Data, addr, size)))
        return;
    }
    CurPtr += COFFHeader->SizeOfOptionalHeader;
  }

  if (!COFFHeader->isImportLibrary())
    if ((ec = getObject(SectionTable, Data, base() + CurPtr,
                        COFFHeader->NumberOfSections * sizeof(coff_section))))
      return;

  // Initialize the pointer to the symbol table.
  if (COFFHeader->PointerToSymbolTable != 0)
    if ((ec = initSymbolTablePtr()))
      return;

  // Initialize the pointer to the beginning of the import table.
  if ((ec = initImportTablePtr()))
    return;

  // Initialize the pointer to the export table.
  if ((ec = initExportTablePtr()))
    return;

  ec = object_error::success;
}

symbol_iterator COFFObjectFile::begin_symbols() const {
  DataRefImpl ret;
  ret.p = reinterpret_cast<uintptr_t>(SymbolTable);
  return symbol_iterator(SymbolRef(ret, this));
}

symbol_iterator COFFObjectFile::end_symbols() const {
  // The symbol table ends where the string table begins.
  DataRefImpl ret;
  ret.p = reinterpret_cast<uintptr_t>(StringTable);
  return symbol_iterator(SymbolRef(ret, this));
}

symbol_iterator COFFObjectFile::begin_dynamic_symbols() const {
  // TODO: implement
  report_fatal_error("Dynamic symbols unimplemented in COFFObjectFile");
}

symbol_iterator COFFObjectFile::end_dynamic_symbols() const {
  // TODO: implement
  report_fatal_error("Dynamic symbols unimplemented in COFFObjectFile");
}

library_iterator COFFObjectFile::begin_libraries_needed() const {
  // TODO: implement
  report_fatal_error("Libraries needed unimplemented in COFFObjectFile");
}

library_iterator COFFObjectFile::end_libraries_needed() const {
  // TODO: implement
  report_fatal_error("Libraries needed unimplemented in COFFObjectFile");
}

StringRef COFFObjectFile::getLoadName() const {
  // COFF does not have this field.
  return "";
}

import_directory_iterator COFFObjectFile::import_directory_begin() const {
  return import_directory_iterator(
      ImportDirectoryEntryRef(ImportDirectory, 0, this));
}

import_directory_iterator COFFObjectFile::import_directory_end() const {
  return import_directory_iterator(
      ImportDirectoryEntryRef(ImportDirectory, NumberOfImportDirectory, this));
}

export_directory_iterator COFFObjectFile::export_directory_begin() const {
  return export_directory_iterator(
      ExportDirectoryEntryRef(ExportDirectory, 0, this));
}

export_directory_iterator COFFObjectFile::export_directory_end() const {
  if (ExportDirectory == 0)
    return export_directory_iterator(ExportDirectoryEntryRef(0, 0, this));
  ExportDirectoryEntryRef ref(ExportDirectory,
                              ExportDirectory->AddressTableEntries, this);
  return export_directory_iterator(ref);
}

section_iterator COFFObjectFile::begin_sections() const {
  DataRefImpl ret;
  ret.p = reinterpret_cast<uintptr_t>(SectionTable);
  return section_iterator(SectionRef(ret, this));
}

section_iterator COFFObjectFile::end_sections() const {
  DataRefImpl ret;
  int numSections = COFFHeader->isImportLibrary()
      ? 0 : COFFHeader->NumberOfSections;
  ret.p = reinterpret_cast<uintptr_t>(SectionTable + numSections);
  return section_iterator(SectionRef(ret, this));
}

uint8_t COFFObjectFile::getBytesInAddress() const {
  return getArch() == Triple::x86_64 ? 8 : 4;
}

StringRef COFFObjectFile::getFileFormatName() const {
  switch(COFFHeader->Machine) {
  case COFF::IMAGE_FILE_MACHINE_I386:
    return "COFF-i386";
  case COFF::IMAGE_FILE_MACHINE_AMD64:
    return "COFF-x86-64";
  default:
    return "COFF-<unknown arch>";
  }
}

unsigned COFFObjectFile::getArch() const {
  switch(COFFHeader->Machine) {
  case COFF::IMAGE_FILE_MACHINE_I386:
    return Triple::x86;
  case COFF::IMAGE_FILE_MACHINE_AMD64:
    return Triple::x86_64;
  default:
    return Triple::UnknownArch;
  }
}

// This method is kept here because lld uses this. As soon as we make
// lld to use getCOFFHeader, this method will be removed.
error_code COFFObjectFile::getHeader(const coff_file_header *&Res) const {
  return getCOFFHeader(Res);
}

error_code COFFObjectFile::getCOFFHeader(const coff_file_header *&Res) const {
  Res = COFFHeader;
  return object_error::success;
}

error_code COFFObjectFile::getPE32Header(const pe32_header *&Res) const {
  Res = PE32Header;
  return object_error::success;
}

error_code COFFObjectFile::getDataDirectory(uint32_t index,
                                            const data_directory *&Res) const {
  // Error if if there's no data directory or the index is out of range.
  if (!DataDirectory || index > PE32Header->NumberOfRvaAndSize)
    return object_error::parse_failed;
  Res = &DataDirectory[index];
  return object_error::success;
}

error_code COFFObjectFile::getSection(int32_t index,
                                      const coff_section *&Result) const {
  // Check for special index values.
  if (index == COFF::IMAGE_SYM_UNDEFINED ||
      index == COFF::IMAGE_SYM_ABSOLUTE ||
      index == COFF::IMAGE_SYM_DEBUG)
    Result = NULL;
  else if (index > 0 && index <= COFFHeader->NumberOfSections)
    // We already verified the section table data, so no need to check again.
    Result = SectionTable + (index - 1);
  else
    return object_error::parse_failed;
  return object_error::success;
}

error_code COFFObjectFile::getString(uint32_t offset,
                                     StringRef &Result) const {
  if (StringTableSize <= 4)
    // Tried to get a string from an empty string table.
    return object_error::parse_failed;
  if (offset >= StringTableSize)
    return object_error::unexpected_eof;
  Result = StringRef(StringTable + offset);
  return object_error::success;
}

error_code COFFObjectFile::getSymbol(uint32_t index,
                                     const coff_symbol *&Result) const {
  if (index < COFFHeader->NumberOfSymbols)
    Result = SymbolTable + index;
  else
    return object_error::parse_failed;
  return object_error::success;
}

error_code COFFObjectFile::getSymbolName(const coff_symbol *symbol,
                                         StringRef &Res) const {
  // Check for string table entry. First 4 bytes are 0.
  if (symbol->Name.Offset.Zeroes == 0) {
    uint32_t Offset = symbol->Name.Offset.Offset;
    if (error_code ec = getString(Offset, Res))
      return ec;
    return object_error::success;
  }

  if (symbol->Name.ShortName[7] == 0)
    // Null terminated, let ::strlen figure out the length.
    Res = StringRef(symbol->Name.ShortName);
  else
    // Not null terminated, use all 8 bytes.
    Res = StringRef(symbol->Name.ShortName, 8);
  return object_error::success;
}

ArrayRef<uint8_t> COFFObjectFile::getSymbolAuxData(
                                  const coff_symbol *symbol) const {
  const uint8_t *aux = NULL;

  if ( symbol->NumberOfAuxSymbols > 0 ) {
  // AUX data comes immediately after the symbol in COFF
    aux = reinterpret_cast<const uint8_t *>(symbol + 1);
# ifndef NDEBUG
    // Verify that the aux symbol points to a valid entry in the symbol table.
    uintptr_t offset = uintptr_t(aux) - uintptr_t(base());
    if (offset < COFFHeader->PointerToSymbolTable
        || offset >= COFFHeader->PointerToSymbolTable
           + (COFFHeader->NumberOfSymbols * sizeof(coff_symbol)))
      report_fatal_error("Aux Symbol data was outside of symbol table.");

    assert((offset - COFFHeader->PointerToSymbolTable) % sizeof(coff_symbol)
         == 0 && "Aux Symbol data did not point to the beginning of a symbol");
# endif
  }
  return ArrayRef<uint8_t>(aux, symbol->NumberOfAuxSymbols * sizeof(coff_symbol));
}

error_code COFFObjectFile::getSectionName(const coff_section *Sec,
                                          StringRef &Res) const {
  StringRef Name;
  if (Sec->Name[7] == 0)
    // Null terminated, let ::strlen figure out the length.
    Name = Sec->Name;
  else
    // Not null terminated, use all 8 bytes.
    Name = StringRef(Sec->Name, 8);

  // Check for string table entry. First byte is '/'.
  if (Name[0] == '/') {
    uint32_t Offset;
    if (Name.substr(1).getAsInteger(10, Offset))
      return object_error::parse_failed;
    if (error_code ec = getString(Offset, Name))
      return ec;
  }

  Res = Name;
  return object_error::success;
}

error_code COFFObjectFile::getSectionContents(const coff_section *Sec,
                                              ArrayRef<uint8_t> &Res) const {
  // The only thing that we need to verify is that the contents is contained
  // within the file bounds. We don't need to make sure it doesn't cover other
  // data, as there's nothing that says that is not allowed.
  uintptr_t ConStart = uintptr_t(base()) + Sec->PointerToRawData;
  uintptr_t ConEnd = ConStart + Sec->SizeOfRawData;
  if (ConEnd > uintptr_t(Data->getBufferEnd()))
    return object_error::parse_failed;
  Res = ArrayRef<uint8_t>(reinterpret_cast<const unsigned char*>(ConStart),
                          Sec->SizeOfRawData);
  return object_error::success;
}

const coff_relocation *COFFObjectFile::toRel(DataRefImpl Rel) const {
  return reinterpret_cast<const coff_relocation*>(Rel.p);
}
error_code COFFObjectFile::getRelocationNext(DataRefImpl Rel,
                                             RelocationRef &Res) const {
  Rel.p = reinterpret_cast<uintptr_t>(
            reinterpret_cast<const coff_relocation*>(Rel.p) + 1);
  Res = RelocationRef(Rel, this);
  return object_error::success;
}
error_code COFFObjectFile::getRelocationAddress(DataRefImpl Rel,
                                                uint64_t &Res) const {
  report_fatal_error("getRelocationAddress not implemented in COFFObjectFile");
}
error_code COFFObjectFile::getRelocationOffset(DataRefImpl Rel,
                                               uint64_t &Res) const {
  Res = toRel(Rel)->VirtualAddress;
  return object_error::success;
}
symbol_iterator COFFObjectFile::getRelocationSymbol(DataRefImpl Rel) const {
  const coff_relocation* R = toRel(Rel);
  DataRefImpl Symb;
  Symb.p = reinterpret_cast<uintptr_t>(SymbolTable + R->SymbolTableIndex);
  return symbol_iterator(SymbolRef(Symb, this));
}
error_code COFFObjectFile::getRelocationType(DataRefImpl Rel,
                                             uint64_t &Res) const {
  const coff_relocation* R = toRel(Rel);
  Res = R->Type;
  return object_error::success;
}

const coff_section *COFFObjectFile::getCOFFSection(section_iterator &It) const {
  return toSec(It->getRawDataRefImpl());
}

const coff_symbol *COFFObjectFile::getCOFFSymbol(symbol_iterator &It) const {
  return toSymb(It->getRawDataRefImpl());
}

const coff_relocation *COFFObjectFile::getCOFFRelocation(
                                             relocation_iterator &It) const {
  return toRel(It->getRawDataRefImpl());
}

#define LLVM_COFF_SWITCH_RELOC_TYPE_NAME(enum) \
  case COFF::enum: res = #enum; break;

error_code COFFObjectFile::getRelocationTypeName(DataRefImpl Rel,
                                          SmallVectorImpl<char> &Result) const {
  const coff_relocation *reloc = toRel(Rel);
  StringRef res;
  switch (COFFHeader->Machine) {
  case COFF::IMAGE_FILE_MACHINE_AMD64:
    switch (reloc->Type) {
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_ABSOLUTE);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_ADDR64);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_ADDR32);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_ADDR32NB);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_REL32);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_REL32_1);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_REL32_2);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_REL32_3);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_REL32_4);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_REL32_5);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_SECTION);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_SECREL);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_SECREL7);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_TOKEN);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_SREL32);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_PAIR);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_AMD64_SSPAN32);
    default:
      res = "Unknown";
    }
    break;
  case COFF::IMAGE_FILE_MACHINE_I386:
    switch (reloc->Type) {
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_I386_ABSOLUTE);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_I386_DIR16);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_I386_REL16);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_I386_DIR32);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_I386_DIR32NB);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_I386_SEG12);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_I386_SECTION);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_I386_SECREL);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_I386_TOKEN);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_I386_SECREL7);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_I386_REL32);
    default:
      res = "Unknown";
    }
    break;
  default:
    res = "Unknown";
  }
  Result.append(res.begin(), res.end());
  return object_error::success;
}

#undef LLVM_COFF_SWITCH_RELOC_TYPE_NAME

error_code COFFObjectFile::getRelocationValueString(DataRefImpl Rel,
                                          SmallVectorImpl<char> &Result) const {
  const coff_relocation *reloc = toRel(Rel);
  const coff_symbol *symb = 0;
  if (error_code ec = getSymbol(reloc->SymbolTableIndex, symb)) return ec;
  DataRefImpl sym;
  sym.p = reinterpret_cast<uintptr_t>(symb);
  StringRef symname;
  if (error_code ec = getSymbolName(sym, symname)) return ec;
  Result.append(symname.begin(), symname.end());
  return object_error::success;
}

error_code COFFObjectFile::getLibraryNext(DataRefImpl LibData,
                                          LibraryRef &Result) const {
  report_fatal_error("getLibraryNext not implemented in COFFObjectFile");
}

error_code COFFObjectFile::getLibraryPath(DataRefImpl LibData,
                                          StringRef &Result) const {
  report_fatal_error("getLibraryPath not implemented in COFFObjectFile");
}

bool ImportDirectoryEntryRef::
operator==(const ImportDirectoryEntryRef &Other) const {
  return ImportTable == Other.ImportTable && Index == Other.Index;
}

error_code
ImportDirectoryEntryRef::getNext(ImportDirectoryEntryRef &Result) const {
  Result = ImportDirectoryEntryRef(ImportTable, Index + 1, OwningObject);
  return object_error::success;
}

error_code ImportDirectoryEntryRef::
getImportTableEntry(const import_directory_table_entry *&Result) const {
  Result = ImportTable;
  return object_error::success;
}

error_code ImportDirectoryEntryRef::getName(StringRef &Result) const {
  uintptr_t IntPtr = 0;
  if (error_code EC = OwningObject->getRvaPtr(ImportTable->NameRVA, IntPtr))
    return EC;
  Result = StringRef(reinterpret_cast<const char *>(IntPtr));
  return object_error::success;
}

error_code ImportDirectoryEntryRef::getImportLookupEntry(
    const import_lookup_table_entry32 *&Result) const {
  uintptr_t IntPtr = 0;
  if (error_code EC =
          OwningObject->getRvaPtr(ImportTable->ImportLookupTableRVA, IntPtr))
    return EC;
  Result = reinterpret_cast<const import_lookup_table_entry32 *>(IntPtr);
  return object_error::success;
}

bool ExportDirectoryEntryRef::
operator==(const ExportDirectoryEntryRef &Other) const {
  return ExportTable == Other.ExportTable && Index == Other.Index;
}

error_code
ExportDirectoryEntryRef::getNext(ExportDirectoryEntryRef &Result) const {
  Result = ExportDirectoryEntryRef(ExportTable, Index + 1, OwningObject);
  return object_error::success;
}

// Returns the export ordinal of the current export symbol.
error_code ExportDirectoryEntryRef::getOrdinal(uint32_t &Result) const {
  Result = ExportTable->OrdinalBase + Index;
  return object_error::success;
}

// Returns the address of the current export symbol.
error_code ExportDirectoryEntryRef::getExportRVA(uint32_t &Result) const {
  uintptr_t IntPtr = 0;
  if (error_code EC = OwningObject->getRvaPtr(
          ExportTable->ExportAddressTableRVA, IntPtr))
    return EC;
  const export_address_table_entry *entry = reinterpret_cast<const export_address_table_entry *>(IntPtr);
  Result = entry[Index].ExportRVA;
  return object_error::success;
}

// Returns the name of the current export symbol. If the symbol is exported only
// by ordinal, the empty string is set as a result.
error_code ExportDirectoryEntryRef::getName(StringRef &Result) const {
  uintptr_t IntPtr = 0;
  if (error_code EC = OwningObject->getRvaPtr(
          ExportTable->OrdinalTableRVA, IntPtr))
    return EC;
  const ulittle16_t *Start = reinterpret_cast<const ulittle16_t *>(IntPtr);

  uint32_t NumEntries = ExportTable->NumberOfNamePointers;
  int Offset = 0;
  for (const ulittle16_t *I = Start, *E = Start + NumEntries;
       I < E; ++I, ++Offset) {
    if (*I != Index)
      continue;
    if (error_code EC = OwningObject->getRvaPtr(
            ExportTable->NamePointerRVA, IntPtr))
      return EC;
    const ulittle32_t *NamePtr = reinterpret_cast<const ulittle32_t *>(IntPtr);
    if (error_code EC = OwningObject->getRvaPtr(NamePtr[Offset], IntPtr))
      return EC;
    Result = StringRef(reinterpret_cast<const char *>(IntPtr));
    return object_error::success;
  }
  Result = "";
  return object_error::success;
}

namespace llvm {
ObjectFile *ObjectFile::createCOFFObjectFile(MemoryBuffer *Object) {
  error_code ec;
  return new COFFObjectFile(Object, ec);
}
} // end namespace llvm
