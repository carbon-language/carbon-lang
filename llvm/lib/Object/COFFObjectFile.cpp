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
#include "llvm/Support/COFF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
#include <limits>

using namespace llvm;
using namespace object;

using support::ulittle8_t;
using support::ulittle16_t;
using support::ulittle32_t;
using support::little16_t;

// Returns false if size is greater than the buffer size. And sets ec.
static bool checkSize(const MemoryBuffer *M, error_code &EC, uint64_t Size) {
  if (M->getBufferSize() < Size) {
    EC = object_error::unexpected_eof;
    return false;
  }
  return true;
}

// Sets Obj unless any bytes in [addr, addr + size) fall outsize of m.
// Returns unexpected_eof if error.
template<typename T>
static error_code getObject(const T *&Obj, const MemoryBuffer *M,
                            const uint8_t *Ptr, const size_t Size = sizeof(T)) {
  uintptr_t Addr = uintptr_t(Ptr);
  if (Addr + Size < Addr ||
      Addr + Size < Size ||
      Addr + Size > uintptr_t(M->getBufferEnd())) {
    return object_error::unexpected_eof;
  }
  Obj = reinterpret_cast<const T *>(Addr);
  return object_error::success;
}

// Decode a string table entry in base 64 (//AAAAAA). Expects \arg Str without
// prefixed slashes.
static bool decodeBase64StringEntry(StringRef Str, uint32_t &Result) {
  assert(Str.size() <= 6 && "String too long, possible overflow.");
  if (Str.size() > 6)
    return true;

  uint64_t Value = 0;
  while (!Str.empty()) {
    unsigned CharVal;
    if (Str[0] >= 'A' && Str[0] <= 'Z') // 0..25
      CharVal = Str[0] - 'A';
    else if (Str[0] >= 'a' && Str[0] <= 'z') // 26..51
      CharVal = Str[0] - 'a' + 26;
    else if (Str[0] >= '0' && Str[0] <= '9') // 52..61
      CharVal = Str[0] - '0' + 52;
    else if (Str[0] == '+') // 62
      CharVal = 62;
    else if (Str[0] == '/') // 63
      CharVal = 63;
    else
      return true;

    Value = (Value * 64) + CharVal;
    Str = Str.substr(1);
  }

  if (Value > std::numeric_limits<uint32_t>::max())
    return true;

  Result = static_cast<uint32_t>(Value);
  return false;
}

const coff_symbol *COFFObjectFile::toSymb(DataRefImpl Ref) const {
  const coff_symbol *Addr = reinterpret_cast<const coff_symbol*>(Ref.p);

# ifndef NDEBUG
  // Verify that the symbol points to a valid entry in the symbol table.
  uintptr_t Offset = uintptr_t(Addr) - uintptr_t(base());
  if (Offset < COFFHeader->PointerToSymbolTable
      || Offset >= COFFHeader->PointerToSymbolTable
         + (COFFHeader->NumberOfSymbols * sizeof(coff_symbol)))
    report_fatal_error("Symbol was outside of symbol table.");

  assert((Offset - COFFHeader->PointerToSymbolTable) % sizeof(coff_symbol)
         == 0 && "Symbol did not point to the beginning of a symbol");
# endif

  return Addr;
}

const coff_section *COFFObjectFile::toSec(DataRefImpl Ref) const {
  const coff_section *Addr = reinterpret_cast<const coff_section*>(Ref.p);

# ifndef NDEBUG
  // Verify that the section points to a valid entry in the section table.
  if (Addr < SectionTable
      || Addr >= (SectionTable + COFFHeader->NumberOfSections))
    report_fatal_error("Section was outside of section table.");

  uintptr_t Offset = uintptr_t(Addr) - uintptr_t(SectionTable);
  assert(Offset % sizeof(coff_section) == 0 &&
         "Section did not point to the beginning of a section");
# endif

  return Addr;
}

void COFFObjectFile::moveSymbolNext(DataRefImpl &Ref) const {
  const coff_symbol *Symb = toSymb(Ref);
  Symb += 1 + Symb->NumberOfAuxSymbols;
  Ref.p = reinterpret_cast<uintptr_t>(Symb);
}

error_code COFFObjectFile::getSymbolName(DataRefImpl Ref,
                                         StringRef &Result) const {
  const coff_symbol *Symb = toSymb(Ref);
  return getSymbolName(Symb, Result);
}

error_code COFFObjectFile::getSymbolFileOffset(DataRefImpl Ref,
                                            uint64_t &Result) const {
  const coff_symbol *Symb = toSymb(Ref);
  const coff_section *Section = NULL;
  if (error_code EC = getSection(Symb->SectionNumber, Section))
    return EC;

  if (Symb->SectionNumber == COFF::IMAGE_SYM_UNDEFINED)
    Result = UnknownAddressOrSize;
  else if (Section)
    Result = Section->PointerToRawData + Symb->Value;
  else
    Result = Symb->Value;
  return object_error::success;
}

error_code COFFObjectFile::getSymbolAddress(DataRefImpl Ref,
                                            uint64_t &Result) const {
  const coff_symbol *Symb = toSymb(Ref);
  const coff_section *Section = NULL;
  if (error_code EC = getSection(Symb->SectionNumber, Section))
    return EC;

  if (Symb->SectionNumber == COFF::IMAGE_SYM_UNDEFINED)
    Result = UnknownAddressOrSize;
  else if (Section)
    Result = Section->VirtualAddress + Symb->Value;
  else
    Result = Symb->Value;
  return object_error::success;
}

error_code COFFObjectFile::getSymbolType(DataRefImpl Ref,
                                         SymbolRef::Type &Result) const {
  const coff_symbol *Symb = toSymb(Ref);
  Result = SymbolRef::ST_Other;
  if (Symb->StorageClass == COFF::IMAGE_SYM_CLASS_EXTERNAL &&
      Symb->SectionNumber == COFF::IMAGE_SYM_UNDEFINED) {
    Result = SymbolRef::ST_Unknown;
  } else if (Symb->getComplexType() == COFF::IMAGE_SYM_DTYPE_FUNCTION) {
    Result = SymbolRef::ST_Function;
  } else {
    uint32_t Characteristics = 0;
    if (!COFF::isReservedSectionNumber(Symb->SectionNumber)) {
      const coff_section *Section = NULL;
      if (error_code EC = getSection(Symb->SectionNumber, Section))
        return EC;
      Characteristics = Section->Characteristics;
    }
    if (Characteristics & COFF::IMAGE_SCN_MEM_READ &&
        ~Characteristics & COFF::IMAGE_SCN_MEM_WRITE) // Read only.
      Result = SymbolRef::ST_Data;
  }
  return object_error::success;
}

uint32_t COFFObjectFile::getSymbolFlags(DataRefImpl Ref) const {
  const coff_symbol *Symb = toSymb(Ref);
  uint32_t Result = SymbolRef::SF_None;

  // TODO: Correctly set SF_FormatSpecific, SF_Common

  if (Symb->SectionNumber == COFF::IMAGE_SYM_UNDEFINED) {
    if (Symb->Value == 0)
      Result |= SymbolRef::SF_Undefined;
    else
      Result |= SymbolRef::SF_Common;
  }


  // TODO: This are certainly too restrictive.
  if (Symb->StorageClass == COFF::IMAGE_SYM_CLASS_EXTERNAL)
    Result |= SymbolRef::SF_Global;

  if (Symb->StorageClass == COFF::IMAGE_SYM_CLASS_WEAK_EXTERNAL)
    Result |= SymbolRef::SF_Weak;

  if (Symb->SectionNumber == COFF::IMAGE_SYM_ABSOLUTE)
    Result |= SymbolRef::SF_Absolute;

  return Result;
}

error_code COFFObjectFile::getSymbolSize(DataRefImpl Ref,
                                         uint64_t &Result) const {
  // FIXME: Return the correct size. This requires looking at all the symbols
  //        in the same section as this symbol, and looking for either the next
  //        symbol, or the end of the section.
  const coff_symbol *Symb = toSymb(Ref);
  const coff_section *Section = NULL;
  if (error_code EC = getSection(Symb->SectionNumber, Section))
    return EC;

  if (Symb->SectionNumber == COFF::IMAGE_SYM_UNDEFINED)
    Result = UnknownAddressOrSize;
  else if (Section)
    Result = Section->SizeOfRawData - Symb->Value;
  else
    Result = 0;
  return object_error::success;
}

error_code COFFObjectFile::getSymbolSection(DataRefImpl Ref,
                                            section_iterator &Result) const {
  const coff_symbol *Symb = toSymb(Ref);
  if (COFF::isReservedSectionNumber(Symb->SectionNumber)) {
    Result = section_end();
  } else {
    const coff_section *Sec = 0;
    if (error_code EC = getSection(Symb->SectionNumber, Sec)) return EC;
    DataRefImpl Ref;
    Ref.p = reinterpret_cast<uintptr_t>(Sec);
    Result = section_iterator(SectionRef(Ref, this));
  }
  return object_error::success;
}

error_code COFFObjectFile::getSymbolValue(DataRefImpl Ref,
                                          uint64_t &Val) const {
  report_fatal_error("getSymbolValue unimplemented in COFFObjectFile");
}

void COFFObjectFile::moveSectionNext(DataRefImpl &Ref) const {
  const coff_section *Sec = toSec(Ref);
  Sec += 1;
  Ref.p = reinterpret_cast<uintptr_t>(Sec);
}

error_code COFFObjectFile::getSectionName(DataRefImpl Ref,
                                          StringRef &Result) const {
  const coff_section *Sec = toSec(Ref);
  return getSectionName(Sec, Result);
}

error_code COFFObjectFile::getSectionAddress(DataRefImpl Ref,
                                             uint64_t &Result) const {
  const coff_section *Sec = toSec(Ref);
  Result = Sec->VirtualAddress;
  return object_error::success;
}

error_code COFFObjectFile::getSectionSize(DataRefImpl Ref,
                                          uint64_t &Result) const {
  const coff_section *Sec = toSec(Ref);
  Result = Sec->SizeOfRawData;
  return object_error::success;
}

error_code COFFObjectFile::getSectionContents(DataRefImpl Ref,
                                              StringRef &Result) const {
  const coff_section *Sec = toSec(Ref);
  ArrayRef<uint8_t> Res;
  error_code EC = getSectionContents(Sec, Res);
  Result = StringRef(reinterpret_cast<const char*>(Res.data()), Res.size());
  return EC;
}

error_code COFFObjectFile::getSectionAlignment(DataRefImpl Ref,
                                               uint64_t &Res) const {
  const coff_section *Sec = toSec(Ref);
  if (!Sec)
    return object_error::parse_failed;
  Res = uint64_t(1) << (((Sec->Characteristics & 0x00F00000) >> 20) - 1);
  return object_error::success;
}

error_code COFFObjectFile::isSectionText(DataRefImpl Ref,
                                         bool &Result) const {
  const coff_section *Sec = toSec(Ref);
  Result = Sec->Characteristics & COFF::IMAGE_SCN_CNT_CODE;
  return object_error::success;
}

error_code COFFObjectFile::isSectionData(DataRefImpl Ref,
                                         bool &Result) const {
  const coff_section *Sec = toSec(Ref);
  Result = Sec->Characteristics & COFF::IMAGE_SCN_CNT_INITIALIZED_DATA;
  return object_error::success;
}

error_code COFFObjectFile::isSectionBSS(DataRefImpl Ref,
                                        bool &Result) const {
  const coff_section *Sec = toSec(Ref);
  Result = Sec->Characteristics & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA;
  return object_error::success;
}

error_code COFFObjectFile::isSectionRequiredForExecution(DataRefImpl Ref,
                                                         bool &Result) const {
  // FIXME: Unimplemented
  Result = true;
  return object_error::success;
}

error_code COFFObjectFile::isSectionVirtual(DataRefImpl Ref,
                                           bool &Result) const {
  const coff_section *Sec = toSec(Ref);
  Result = Sec->Characteristics & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA;
  return object_error::success;
}

error_code COFFObjectFile::isSectionZeroInit(DataRefImpl Ref,
                                             bool &Result) const {
  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

error_code COFFObjectFile::isSectionReadOnlyData(DataRefImpl Ref,
                                                bool &Result) const {
  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

error_code COFFObjectFile::sectionContainsSymbol(DataRefImpl SecRef,
                                                 DataRefImpl SymbRef,
                                                 bool &Result) const {
  const coff_section *Sec = toSec(SecRef);
  const coff_symbol *Symb = toSymb(SymbRef);
  const coff_section *SymbSec = 0;
  if (error_code EC = getSection(Symb->SectionNumber, SymbSec)) return EC;
  if (SymbSec == Sec)
    Result = true;
  else
    Result = false;
  return object_error::success;
}

relocation_iterator COFFObjectFile::section_rel_begin(DataRefImpl Ref) const {
  const coff_section *Sec = toSec(Ref);
  DataRefImpl Ret;
  if (Sec->NumberOfRelocations == 0)
    Ret.p = 0;
  else
    Ret.p = reinterpret_cast<uintptr_t>(base() + Sec->PointerToRelocations);

  return relocation_iterator(RelocationRef(Ret, this));
}

relocation_iterator COFFObjectFile::section_rel_end(DataRefImpl Ref) const {
  const coff_section *Sec = toSec(Ref);
  DataRefImpl Ret;
  if (Sec->NumberOfRelocations == 0)
    Ret.p = 0;
  else
    Ret.p = reinterpret_cast<uintptr_t>(
              reinterpret_cast<const coff_relocation*>(
                base() + Sec->PointerToRelocations)
              + Sec->NumberOfRelocations);

  return relocation_iterator(RelocationRef(Ret, this));
}

// Initialize the pointer to the symbol table.
error_code COFFObjectFile::initSymbolTablePtr() {
  if (error_code EC = getObject(
          SymbolTable, Data, base() + COFFHeader->PointerToSymbolTable,
          COFFHeader->NumberOfSymbols * sizeof(coff_symbol)))
    return EC;

  // Find string table. The first four byte of the string table contains the
  // total size of the string table, including the size field itself. If the
  // string table is empty, the value of the first four byte would be 4.
  const uint8_t *StringTableAddr =
      base() + COFFHeader->PointerToSymbolTable +
      COFFHeader->NumberOfSymbols * sizeof(coff_symbol);
  const ulittle32_t *StringTableSizePtr;
  if (error_code EC = getObject(StringTableSizePtr, Data, StringTableAddr))
    return EC;
  StringTableSize = *StringTableSizePtr;
  if (error_code EC =
      getObject(StringTable, Data, StringTableAddr, StringTableSize))
    return EC;

  // Treat table sizes < 4 as empty because contrary to the PECOFF spec, some
  // tools like cvtres write a size of 0 for an empty table instead of 4.
  if (StringTableSize < 4)
      StringTableSize = 4;

  // Check that the string table is null terminated if has any in it.
  if (StringTableSize > 4 && StringTable[StringTableSize - 1] != 0)
    return  object_error::parse_failed;
  return object_error::success;
}

// Returns the file offset for the given VA.
error_code COFFObjectFile::getVaPtr(uint64_t Addr, uintptr_t &Res) const {
  uint64_t ImageBase = PE32Header ? (uint64_t)PE32Header->ImageBase
                                  : (uint64_t)PE32PlusHeader->ImageBase;
  uint64_t Rva = Addr - ImageBase;
  assert(Rva <= UINT32_MAX);
  return getRvaPtr((uint32_t)Rva, Res);
}

// Returns the file offset for the given RVA.
error_code COFFObjectFile::getRvaPtr(uint32_t Addr, uintptr_t &Res) const {
  for (const SectionRef &S : sections()) {
    const coff_section *Section = getCOFFSection(S);
    uint32_t SectionStart = Section->VirtualAddress;
    uint32_t SectionEnd = Section->VirtualAddress + Section->VirtualSize;
    if (SectionStart <= Addr && Addr < SectionEnd) {
      uint32_t Offset = Addr - SectionStart;
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
  if (error_code EC = getRvaPtr(Rva, IntPtr))
    return EC;
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
  if (error_code EC = getRvaPtr(ImportTableRva, IntPtr))
    return EC;
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
  ExportDirectory =
      reinterpret_cast<const export_directory_table_entry *>(IntPtr);
  return object_error::success;
}

COFFObjectFile::COFFObjectFile(MemoryBuffer *Object, error_code &EC,
                               bool BufferOwned)
    : ObjectFile(Binary::ID_COFF, Object, BufferOwned), COFFHeader(0),
      PE32Header(0), PE32PlusHeader(0), DataDirectory(0), SectionTable(0),
      SymbolTable(0), StringTable(0), StringTableSize(0), ImportDirectory(0),
      NumberOfImportDirectory(0), ExportDirectory(0) {
  // Check that we at least have enough room for a header.
  if (!checkSize(Data, EC, sizeof(coff_file_header))) return;

  // The current location in the file where we are looking at.
  uint64_t CurPtr = 0;

  // PE header is optional and is present only in executables. If it exists,
  // it is placed right after COFF header.
  bool HasPEHeader = false;

  // Check if this is a PE/COFF file.
  if (base()[0] == 0x4d && base()[1] == 0x5a) {
    // PE/COFF, seek through MS-DOS compatibility stub and 4-byte
    // PE signature to find 'normal' COFF header.
    if (!checkSize(Data, EC, 0x3c + 8)) return;
    CurPtr = *reinterpret_cast<const ulittle16_t *>(base() + 0x3c);
    // Check the PE magic bytes. ("PE\0\0")
    if (std::memcmp(base() + CurPtr, "PE\0\0", 4) != 0) {
      EC = object_error::parse_failed;
      return;
    }
    CurPtr += 4; // Skip the PE magic bytes.
    HasPEHeader = true;
  }

  if ((EC = getObject(COFFHeader, Data, base() + CurPtr)))
    return;
  CurPtr += sizeof(coff_file_header);

  if (HasPEHeader) {
    const pe32_header *Header;
    if ((EC = getObject(Header, Data, base() + CurPtr)))
      return;

    const uint8_t *DataDirAddr;
    uint64_t DataDirSize;
    if (Header->Magic == 0x10b) {
      PE32Header = Header;
      DataDirAddr = base() + CurPtr + sizeof(pe32_header);
      DataDirSize = sizeof(data_directory) * PE32Header->NumberOfRvaAndSize;
    } else if (Header->Magic == 0x20b) {
      PE32PlusHeader = reinterpret_cast<const pe32plus_header *>(Header);
      DataDirAddr = base() + CurPtr + sizeof(pe32plus_header);
      DataDirSize = sizeof(data_directory) * PE32PlusHeader->NumberOfRvaAndSize;
    } else {
      // It's neither PE32 nor PE32+.
      EC = object_error::parse_failed;
      return;
    }
    if ((EC = getObject(DataDirectory, Data, DataDirAddr, DataDirSize)))
      return;
    CurPtr += COFFHeader->SizeOfOptionalHeader;
  }

  if (COFFHeader->isImportLibrary())
    return;

  if ((EC = getObject(SectionTable, Data, base() + CurPtr,
                      COFFHeader->NumberOfSections * sizeof(coff_section))))
    return;

  // Initialize the pointer to the symbol table.
  if (COFFHeader->PointerToSymbolTable != 0)
    if ((EC = initSymbolTablePtr()))
      return;

  // Initialize the pointer to the beginning of the import table.
  if ((EC = initImportTablePtr()))
    return;

  // Initialize the pointer to the export table.
  if ((EC = initExportTablePtr()))
    return;

  EC = object_error::success;
}

basic_symbol_iterator COFFObjectFile::symbol_begin_impl() const {
  DataRefImpl Ret;
  Ret.p = reinterpret_cast<uintptr_t>(SymbolTable);
  return basic_symbol_iterator(SymbolRef(Ret, this));
}

basic_symbol_iterator COFFObjectFile::symbol_end_impl() const {
  // The symbol table ends where the string table begins.
  DataRefImpl Ret;
  Ret.p = reinterpret_cast<uintptr_t>(StringTable);
  return basic_symbol_iterator(SymbolRef(Ret, this));
}

library_iterator COFFObjectFile::needed_library_begin() const {
  // TODO: implement
  report_fatal_error("Libraries needed unimplemented in COFFObjectFile");
}

library_iterator COFFObjectFile::needed_library_end() const {
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
  ExportDirectoryEntryRef Ref(ExportDirectory,
                              ExportDirectory->AddressTableEntries, this);
  return export_directory_iterator(Ref);
}

section_iterator COFFObjectFile::section_begin() const {
  DataRefImpl Ret;
  Ret.p = reinterpret_cast<uintptr_t>(SectionTable);
  return section_iterator(SectionRef(Ret, this));
}

section_iterator COFFObjectFile::section_end() const {
  DataRefImpl Ret;
  int NumSections = COFFHeader->isImportLibrary()
      ? 0 : COFFHeader->NumberOfSections;
  Ret.p = reinterpret_cast<uintptr_t>(SectionTable + NumSections);
  return section_iterator(SectionRef(Ret, this));
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
  case COFF::IMAGE_FILE_MACHINE_ARMNT:
    return "COFF-ARM";
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
  case COFF::IMAGE_FILE_MACHINE_ARMNT:
    return Triple::thumb;
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

error_code
COFFObjectFile::getPE32PlusHeader(const pe32plus_header *&Res) const {
  Res = PE32PlusHeader;
  return object_error::success;
}

error_code COFFObjectFile::getDataDirectory(uint32_t Index,
                                            const data_directory *&Res) const {
  // Error if if there's no data directory or the index is out of range.
  if (!DataDirectory)
    return object_error::parse_failed;
  assert(PE32Header || PE32PlusHeader);
  uint32_t NumEnt = PE32Header ? PE32Header->NumberOfRvaAndSize
                               : PE32PlusHeader->NumberOfRvaAndSize;
  if (Index > NumEnt)
    return object_error::parse_failed;
  Res = &DataDirectory[Index];
  return object_error::success;
}

error_code COFFObjectFile::getSection(int32_t Index,
                                      const coff_section *&Result) const {
  // Check for special index values.
  if (COFF::isReservedSectionNumber(Index))
    Result = NULL;
  else if (Index > 0 && Index <= COFFHeader->NumberOfSections)
    // We already verified the section table data, so no need to check again.
    Result = SectionTable + (Index - 1);
  else
    return object_error::parse_failed;
  return object_error::success;
}

error_code COFFObjectFile::getString(uint32_t Offset,
                                     StringRef &Result) const {
  if (StringTableSize <= 4)
    // Tried to get a string from an empty string table.
    return object_error::parse_failed;
  if (Offset >= StringTableSize)
    return object_error::unexpected_eof;
  Result = StringRef(StringTable + Offset);
  return object_error::success;
}

error_code COFFObjectFile::getSymbol(uint32_t Index,
                                     const coff_symbol *&Result) const {
  if (Index < COFFHeader->NumberOfSymbols)
    Result = SymbolTable + Index;
  else
    return object_error::parse_failed;
  return object_error::success;
}

error_code COFFObjectFile::getSymbolName(const coff_symbol *Symbol,
                                         StringRef &Res) const {
  // Check for string table entry. First 4 bytes are 0.
  if (Symbol->Name.Offset.Zeroes == 0) {
    uint32_t Offset = Symbol->Name.Offset.Offset;
    if (error_code EC = getString(Offset, Res))
      return EC;
    return object_error::success;
  }

  if (Symbol->Name.ShortName[7] == 0)
    // Null terminated, let ::strlen figure out the length.
    Res = StringRef(Symbol->Name.ShortName);
  else
    // Not null terminated, use all 8 bytes.
    Res = StringRef(Symbol->Name.ShortName, 8);
  return object_error::success;
}

ArrayRef<uint8_t> COFFObjectFile::getSymbolAuxData(
                                  const coff_symbol *Symbol) const {
  const uint8_t *Aux = NULL;

  if (Symbol->NumberOfAuxSymbols > 0) {
  // AUX data comes immediately after the symbol in COFF
    Aux = reinterpret_cast<const uint8_t *>(Symbol + 1);
# ifndef NDEBUG
    // Verify that the Aux symbol points to a valid entry in the symbol table.
    uintptr_t Offset = uintptr_t(Aux) - uintptr_t(base());
    if (Offset < COFFHeader->PointerToSymbolTable
        || Offset >= COFFHeader->PointerToSymbolTable
           + (COFFHeader->NumberOfSymbols * sizeof(coff_symbol)))
      report_fatal_error("Aux Symbol data was outside of symbol table.");

    assert((Offset - COFFHeader->PointerToSymbolTable) % sizeof(coff_symbol)
         == 0 && "Aux Symbol data did not point to the beginning of a symbol");
# endif
  }
  return ArrayRef<uint8_t>(Aux,
                           Symbol->NumberOfAuxSymbols * sizeof(coff_symbol));
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
    if (Name[1] == '/') {
      if (decodeBase64StringEntry(Name.substr(2), Offset))
        return object_error::parse_failed;
    } else {
      if (Name.substr(1).getAsInteger(10, Offset))
        return object_error::parse_failed;
    }
    if (error_code EC = getString(Offset, Name))
      return EC;
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

void COFFObjectFile::moveRelocationNext(DataRefImpl &Rel) const {
  Rel.p = reinterpret_cast<uintptr_t>(
            reinterpret_cast<const coff_relocation*>(Rel.p) + 1);
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
  DataRefImpl Ref;
  Ref.p = reinterpret_cast<uintptr_t>(SymbolTable + R->SymbolTableIndex);
  return symbol_iterator(SymbolRef(Ref, this));
}

error_code COFFObjectFile::getRelocationType(DataRefImpl Rel,
                                             uint64_t &Res) const {
  const coff_relocation* R = toRel(Rel);
  Res = R->Type;
  return object_error::success;
}

const coff_section *
COFFObjectFile::getCOFFSection(const SectionRef &Section) const {
  return toSec(Section.getRawDataRefImpl());
}

const coff_symbol *
COFFObjectFile::getCOFFSymbol(const SymbolRef &Symbol) const {
  return toSymb(Symbol.getRawDataRefImpl());
}

const coff_relocation *
COFFObjectFile::getCOFFRelocation(const RelocationRef &Reloc) const {
  return toRel(Reloc.getRawDataRefImpl());
}

#define LLVM_COFF_SWITCH_RELOC_TYPE_NAME(reloc_type)                           \
  case COFF::reloc_type:                                                       \
    Res = #reloc_type;                                                         \
    break;

error_code COFFObjectFile::getRelocationTypeName(DataRefImpl Rel,
                                          SmallVectorImpl<char> &Result) const {
  const coff_relocation *Reloc = toRel(Rel);
  StringRef Res;
  switch (COFFHeader->Machine) {
  case COFF::IMAGE_FILE_MACHINE_AMD64:
    switch (Reloc->Type) {
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
      Res = "Unknown";
    }
    break;
  case COFF::IMAGE_FILE_MACHINE_I386:
    switch (Reloc->Type) {
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
      Res = "Unknown";
    }
    break;
  default:
    Res = "Unknown";
  }
  Result.append(Res.begin(), Res.end());
  return object_error::success;
}

#undef LLVM_COFF_SWITCH_RELOC_TYPE_NAME

error_code COFFObjectFile::getRelocationValueString(DataRefImpl Rel,
                                          SmallVectorImpl<char> &Result) const {
  const coff_relocation *Reloc = toRel(Rel);
  const coff_symbol *Symb = 0;
  if (error_code EC = getSymbol(Reloc->SymbolTableIndex, Symb)) return EC;
  DataRefImpl Sym;
  Sym.p = reinterpret_cast<uintptr_t>(Symb);
  StringRef SymName;
  if (error_code EC = getSymbolName(Sym, SymName)) return EC;
  Result.append(SymName.begin(), SymName.end());
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

void ImportDirectoryEntryRef::moveNext() {
  ++Index;
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

void ExportDirectoryEntryRef::moveNext() {
  ++Index;
}

// Returns the name of the current export symbol. If the symbol is exported only
// by ordinal, the empty string is set as a result.
error_code ExportDirectoryEntryRef::getDllName(StringRef &Result) const {
  uintptr_t IntPtr = 0;
  if (error_code EC = OwningObject->getRvaPtr(ExportTable->NameRVA, IntPtr))
    return EC;
  Result = StringRef(reinterpret_cast<const char *>(IntPtr));
  return object_error::success;
}

// Returns the starting ordinal number.
error_code ExportDirectoryEntryRef::getOrdinalBase(uint32_t &Result) const {
  Result = ExportTable->OrdinalBase;
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
  const export_address_table_entry *entry =
      reinterpret_cast<const export_address_table_entry *>(IntPtr);
  Result = entry[Index].ExportRVA;
  return object_error::success;
}

// Returns the name of the current export symbol. If the symbol is exported only
// by ordinal, the empty string is set as a result.
error_code ExportDirectoryEntryRef::getSymbolName(StringRef &Result) const {
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

ErrorOr<ObjectFile *> ObjectFile::createCOFFObjectFile(MemoryBuffer *Object,
                                                       bool BufferOwned) {
  error_code EC;
  std::unique_ptr<COFFObjectFile> Ret(
      new COFFObjectFile(Object, EC, BufferOwned));
  if (EC)
    return EC;
  return Ret.release();
}
