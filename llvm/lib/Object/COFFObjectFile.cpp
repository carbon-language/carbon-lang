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

using support::ulittle16_t;
using support::ulittle32_t;
using support::ulittle64_t;
using support::little16_t;

// Returns false if size is greater than the buffer size. And sets ec.
static bool checkSize(MemoryBufferRef M, std::error_code &EC, uint64_t Size) {
  if (M.getBufferSize() < Size) {
    EC = object_error::unexpected_eof;
    return false;
  }
  return true;
}

// Sets Obj unless any bytes in [addr, addr + size) fall outsize of m.
// Returns unexpected_eof if error.
template <typename T>
static std::error_code getObject(const T *&Obj, MemoryBufferRef M,
                                 const uint8_t *Ptr,
                                 const size_t Size = sizeof(T)) {
  uintptr_t Addr = uintptr_t(Ptr);
  if (Addr + Size < Addr || Addr + Size < Size ||
      Addr + Size > uintptr_t(M.getBufferEnd())) {
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

template <typename coff_symbol_type>
const coff_symbol_type *COFFObjectFile::toSymb(DataRefImpl Ref) const {
  const coff_symbol_type *Addr =
      reinterpret_cast<const coff_symbol_type *>(Ref.p);

#ifndef NDEBUG
  // Verify that the symbol points to a valid entry in the symbol table.
  uintptr_t Offset = uintptr_t(Addr) - uintptr_t(base());
  if (Offset < getPointerToSymbolTable() ||
      Offset >= getPointerToSymbolTable() +
                    (getNumberOfSymbols() * sizeof(coff_symbol_type)))
    report_fatal_error("Symbol was outside of symbol table.");

  assert((Offset - getPointerToSymbolTable()) % sizeof(coff_symbol_type) == 0 &&
         "Symbol did not point to the beginning of a symbol");
#endif

  return Addr;
}

const coff_section *COFFObjectFile::toSec(DataRefImpl Ref) const {
  const coff_section *Addr = reinterpret_cast<const coff_section*>(Ref.p);

# ifndef NDEBUG
  // Verify that the section points to a valid entry in the section table.
  if (Addr < SectionTable || Addr >= (SectionTable + getNumberOfSections()))
    report_fatal_error("Section was outside of section table.");

  uintptr_t Offset = uintptr_t(Addr) - uintptr_t(SectionTable);
  assert(Offset % sizeof(coff_section) == 0 &&
         "Section did not point to the beginning of a section");
# endif

  return Addr;
}

void COFFObjectFile::moveSymbolNext(DataRefImpl &Ref) const {
  if (SymbolTable16) {
    const coff_symbol16 *Symb = toSymb<coff_symbol16>(Ref);
    Symb += 1 + Symb->NumberOfAuxSymbols;
    Ref.p = reinterpret_cast<uintptr_t>(Symb);
  } else if (SymbolTable32) {
    const coff_symbol32 *Symb = toSymb<coff_symbol32>(Ref);
    Symb += 1 + Symb->NumberOfAuxSymbols;
    Ref.p = reinterpret_cast<uintptr_t>(Symb);
  } else {
    llvm_unreachable("no symbol table pointer!");
  }
}

std::error_code COFFObjectFile::getSymbolName(DataRefImpl Ref,
                                              StringRef &Result) const {
  COFFSymbolRef Symb = getCOFFSymbol(Ref);
  return getSymbolName(Symb, Result);
}

std::error_code COFFObjectFile::getSymbolAddress(DataRefImpl Ref,
                                                 uint64_t &Result) const {
  COFFSymbolRef Symb = getCOFFSymbol(Ref);

  if (Symb.isAnyUndefined()) {
    Result = UnknownAddressOrSize;
    return object_error::success;
  }
  if (Symb.isCommon()) {
    Result = UnknownAddressOrSize;
    return object_error::success;
  }
  int32_t SectionNumber = Symb.getSectionNumber();
  if (!COFF::isReservedSectionNumber(SectionNumber)) {
    const coff_section *Section = nullptr;
    if (std::error_code EC = getSection(SectionNumber, Section))
      return EC;

    Result = Section->VirtualAddress + Symb.getValue();
    return object_error::success;
  }

  Result = Symb.getValue();
  return object_error::success;
}

std::error_code COFFObjectFile::getSymbolType(DataRefImpl Ref,
                                              SymbolRef::Type &Result) const {
  COFFSymbolRef Symb = getCOFFSymbol(Ref);
  int32_t SectionNumber = Symb.getSectionNumber();
  Result = SymbolRef::ST_Other;

  if (Symb.isAnyUndefined()) {
    Result = SymbolRef::ST_Unknown;
  } else if (Symb.isFunctionDefinition()) {
    Result = SymbolRef::ST_Function;
  } else if (Symb.isCommon()) {
    Result = SymbolRef::ST_Data;
  } else if (Symb.isFileRecord()) {
    Result = SymbolRef::ST_File;
  } else if (SectionNumber == COFF::IMAGE_SYM_DEBUG) {
    Result = SymbolRef::ST_Debug;
  } else if (!COFF::isReservedSectionNumber(SectionNumber)) {
    const coff_section *Section = nullptr;
    if (std::error_code EC = getSection(SectionNumber, Section))
      return EC;
    uint32_t Characteristics = Section->Characteristics;
    if (Characteristics & COFF::IMAGE_SCN_CNT_CODE)
      Result = SymbolRef::ST_Function;
    else if (Characteristics & (COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                                COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA))
      Result = SymbolRef::ST_Data;
  }
  return object_error::success;
}

uint32_t COFFObjectFile::getSymbolFlags(DataRefImpl Ref) const {
  COFFSymbolRef Symb = getCOFFSymbol(Ref);
  uint32_t Result = SymbolRef::SF_None;

  if (Symb.isExternal() || Symb.isWeakExternal())
    Result |= SymbolRef::SF_Global;

  if (Symb.isWeakExternal())
    Result |= SymbolRef::SF_Weak;

  if (Symb.getSectionNumber() == COFF::IMAGE_SYM_ABSOLUTE)
    Result |= SymbolRef::SF_Absolute;

  if (Symb.isFileRecord())
    Result |= SymbolRef::SF_FormatSpecific;

  if (Symb.isSectionDefinition())
    Result |= SymbolRef::SF_FormatSpecific;

  if (Symb.isCommon())
    Result |= SymbolRef::SF_Common;

  if (Symb.isAnyUndefined())
    Result |= SymbolRef::SF_Undefined;

  return Result;
}

std::error_code COFFObjectFile::getSymbolSize(DataRefImpl Ref,
                                              uint64_t &Result) const {
  COFFSymbolRef Symb = getCOFFSymbol(Ref);

  if (Symb.isAnyUndefined()) {
    Result = UnknownAddressOrSize;
    return object_error::success;
  }
  if (Symb.isCommon()) {
    Result = Symb.getValue();
    return object_error::success;
  }
  if (Symb.isFunctionDefinition()) {
    ArrayRef<uint8_t> AuxData = getSymbolAuxData(Symb);
    if (!AuxData.empty()) {
      const auto *CAFD =
          reinterpret_cast<const coff_aux_function_definition *>(
              AuxData.data());
      Result = CAFD->TotalSize;
      return object_error::success;
    }
  }
  // FIXME: Return the correct size. This requires looking at all the symbols
  //        in the same section as this symbol, and looking for either the next
  //        symbol, or the end of the section.
  int32_t SectionNumber = Symb.getSectionNumber();
  if (!COFF::isReservedSectionNumber(SectionNumber)) {
    const coff_section *Section = nullptr;
    if (std::error_code EC = getSection(SectionNumber, Section))
      return EC;

    Result = Section->SizeOfRawData - Symb.getValue();
    return object_error::success;
  }

  Result = 0;
  return object_error::success;
}

std::error_code
COFFObjectFile::getSymbolSection(DataRefImpl Ref,
                                 section_iterator &Result) const {
  COFFSymbolRef Symb = getCOFFSymbol(Ref);
  if (COFF::isReservedSectionNumber(Symb.getSectionNumber())) {
    Result = section_end();
  } else {
    const coff_section *Sec = nullptr;
    if (std::error_code EC = getSection(Symb.getSectionNumber(), Sec))
      return EC;
    DataRefImpl Ref;
    Ref.p = reinterpret_cast<uintptr_t>(Sec);
    Result = section_iterator(SectionRef(Ref, this));
  }
  return object_error::success;
}

void COFFObjectFile::moveSectionNext(DataRefImpl &Ref) const {
  const coff_section *Sec = toSec(Ref);
  Sec += 1;
  Ref.p = reinterpret_cast<uintptr_t>(Sec);
}

std::error_code COFFObjectFile::getSectionName(DataRefImpl Ref,
                                               StringRef &Result) const {
  const coff_section *Sec = toSec(Ref);
  return getSectionName(Sec, Result);
}

uint64_t COFFObjectFile::getSectionAddress(DataRefImpl Ref) const {
  const coff_section *Sec = toSec(Ref);
  return Sec->VirtualAddress;
}

uint64_t COFFObjectFile::getSectionSize(DataRefImpl Ref) const {
  return getSectionSize(toSec(Ref));
}

std::error_code COFFObjectFile::getSectionContents(DataRefImpl Ref,
                                                   StringRef &Result) const {
  const coff_section *Sec = toSec(Ref);
  ArrayRef<uint8_t> Res;
  std::error_code EC = getSectionContents(Sec, Res);
  Result = StringRef(reinterpret_cast<const char*>(Res.data()), Res.size());
  return EC;
}

uint64_t COFFObjectFile::getSectionAlignment(DataRefImpl Ref) const {
  const coff_section *Sec = toSec(Ref);
  return uint64_t(1) << (((Sec->Characteristics & 0x00F00000) >> 20) - 1);
}

bool COFFObjectFile::isSectionText(DataRefImpl Ref) const {
  const coff_section *Sec = toSec(Ref);
  return Sec->Characteristics & COFF::IMAGE_SCN_CNT_CODE;
}

bool COFFObjectFile::isSectionData(DataRefImpl Ref) const {
  const coff_section *Sec = toSec(Ref);
  return Sec->Characteristics & COFF::IMAGE_SCN_CNT_INITIALIZED_DATA;
}

bool COFFObjectFile::isSectionBSS(DataRefImpl Ref) const {
  const coff_section *Sec = toSec(Ref);
  return Sec->Characteristics & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA;
}

bool COFFObjectFile::isSectionRequiredForExecution(DataRefImpl Ref) const {
  // FIXME: Unimplemented
  return true;
}

bool COFFObjectFile::isSectionVirtual(DataRefImpl Ref) const {
  const coff_section *Sec = toSec(Ref);
  return Sec->Characteristics & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA;
}

bool COFFObjectFile::isSectionZeroInit(DataRefImpl Ref) const {
  // FIXME: Unimplemented.
  return false;
}

bool COFFObjectFile::isSectionReadOnlyData(DataRefImpl Ref) const {
  // FIXME: Unimplemented.
  return false;
}

bool COFFObjectFile::sectionContainsSymbol(DataRefImpl SecRef,
                                           DataRefImpl SymbRef) const {
  const coff_section *Sec = toSec(SecRef);
  COFFSymbolRef Symb = getCOFFSymbol(SymbRef);
  int32_t SecNumber = (Sec - SectionTable) + 1;
  return SecNumber == Symb.getSectionNumber();
}

relocation_iterator COFFObjectFile::section_rel_begin(DataRefImpl Ref) const {
  const coff_section *Sec = toSec(Ref);
  DataRefImpl Ret;
  if (Sec->NumberOfRelocations == 0) {
    Ret.p = 0;
  } else {
    auto begin = reinterpret_cast<const coff_relocation*>(
        base() + Sec->PointerToRelocations);
    if (Sec->hasExtendedRelocations()) {
      // Skip the first relocation entry repurposed to store the number of
      // relocations.
      begin++;
    }
    Ret.p = reinterpret_cast<uintptr_t>(begin);
  }
  return relocation_iterator(RelocationRef(Ret, this));
}

static uint32_t getNumberOfRelocations(const coff_section *Sec,
                                       const uint8_t *base) {
  // The field for the number of relocations in COFF section table is only
  // 16-bit wide. If a section has more than 65535 relocations, 0xFFFF is set to
  // NumberOfRelocations field, and the actual relocation count is stored in the
  // VirtualAddress field in the first relocation entry.
  if (Sec->hasExtendedRelocations()) {
    auto *FirstReloc = reinterpret_cast<const coff_relocation*>(
        base + Sec->PointerToRelocations);
    return FirstReloc->VirtualAddress;
  }
  return Sec->NumberOfRelocations;
}

relocation_iterator COFFObjectFile::section_rel_end(DataRefImpl Ref) const {
  const coff_section *Sec = toSec(Ref);
  DataRefImpl Ret;
  if (Sec->NumberOfRelocations == 0) {
    Ret.p = 0;
  } else {
    auto begin = reinterpret_cast<const coff_relocation*>(
        base() + Sec->PointerToRelocations);
    uint32_t NumReloc = getNumberOfRelocations(Sec, base());
    Ret.p = reinterpret_cast<uintptr_t>(begin + NumReloc);
  }
  return relocation_iterator(RelocationRef(Ret, this));
}

// Initialize the pointer to the symbol table.
std::error_code COFFObjectFile::initSymbolTablePtr() {
  if (COFFHeader)
    if (std::error_code EC =
            getObject(SymbolTable16, Data, base() + getPointerToSymbolTable(),
                      getNumberOfSymbols() * getSymbolTableEntrySize()))
      return EC;

  if (COFFBigObjHeader)
    if (std::error_code EC =
            getObject(SymbolTable32, Data, base() + getPointerToSymbolTable(),
                      getNumberOfSymbols() * getSymbolTableEntrySize()))
      return EC;

  // Find string table. The first four byte of the string table contains the
  // total size of the string table, including the size field itself. If the
  // string table is empty, the value of the first four byte would be 4.
  const uint8_t *StringTableAddr =
      base() + getPointerToSymbolTable() +
      getNumberOfSymbols() * getSymbolTableEntrySize();
  const ulittle32_t *StringTableSizePtr;
  if (std::error_code EC = getObject(StringTableSizePtr, Data, StringTableAddr))
    return EC;
  StringTableSize = *StringTableSizePtr;
  if (std::error_code EC =
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
std::error_code COFFObjectFile::getVaPtr(uint64_t Addr, uintptr_t &Res) const {
  uint64_t ImageBase = PE32Header ? (uint64_t)PE32Header->ImageBase
                                  : (uint64_t)PE32PlusHeader->ImageBase;
  uint64_t Rva = Addr - ImageBase;
  assert(Rva <= UINT32_MAX);
  return getRvaPtr((uint32_t)Rva, Res);
}

// Returns the file offset for the given RVA.
std::error_code COFFObjectFile::getRvaPtr(uint32_t Addr, uintptr_t &Res) const {
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
std::error_code COFFObjectFile::getHintName(uint32_t Rva, uint16_t &Hint,
                                            StringRef &Name) const {
  uintptr_t IntPtr = 0;
  if (std::error_code EC = getRvaPtr(Rva, IntPtr))
    return EC;
  const uint8_t *Ptr = reinterpret_cast<const uint8_t *>(IntPtr);
  Hint = *reinterpret_cast<const ulittle16_t *>(Ptr);
  Name = StringRef(reinterpret_cast<const char *>(Ptr + 2));
  return object_error::success;
}

// Find the import table.
std::error_code COFFObjectFile::initImportTablePtr() {
  // First, we get the RVA of the import table. If the file lacks a pointer to
  // the import table, do nothing.
  const data_directory *DataEntry;
  if (getDataDirectory(COFF::IMPORT_TABLE, DataEntry))
    return object_error::success;

  // Do nothing if the pointer to import table is NULL.
  if (DataEntry->RelativeVirtualAddress == 0)
    return object_error::success;

  uint32_t ImportTableRva = DataEntry->RelativeVirtualAddress;
  // -1 because the last entry is the null entry.
  NumberOfImportDirectory = DataEntry->Size /
      sizeof(import_directory_table_entry) - 1;

  // Find the section that contains the RVA. This is needed because the RVA is
  // the import table's memory address which is different from its file offset.
  uintptr_t IntPtr = 0;
  if (std::error_code EC = getRvaPtr(ImportTableRva, IntPtr))
    return EC;
  ImportDirectory = reinterpret_cast<
      const import_directory_table_entry *>(IntPtr);
  return object_error::success;
}

// Initializes DelayImportDirectory and NumberOfDelayImportDirectory.
std::error_code COFFObjectFile::initDelayImportTablePtr() {
  const data_directory *DataEntry;
  if (getDataDirectory(COFF::DELAY_IMPORT_DESCRIPTOR, DataEntry))
    return object_error::success;
  if (DataEntry->RelativeVirtualAddress == 0)
    return object_error::success;

  uint32_t RVA = DataEntry->RelativeVirtualAddress;
  NumberOfDelayImportDirectory = DataEntry->Size /
      sizeof(delay_import_directory_table_entry) - 1;

  uintptr_t IntPtr = 0;
  if (std::error_code EC = getRvaPtr(RVA, IntPtr))
    return EC;
  DelayImportDirectory = reinterpret_cast<
      const delay_import_directory_table_entry *>(IntPtr);
  return object_error::success;
}

// Find the export table.
std::error_code COFFObjectFile::initExportTablePtr() {
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
  if (std::error_code EC = getRvaPtr(ExportTableRva, IntPtr))
    return EC;
  ExportDirectory =
      reinterpret_cast<const export_directory_table_entry *>(IntPtr);
  return object_error::success;
}

COFFObjectFile::COFFObjectFile(MemoryBufferRef Object, std::error_code &EC)
    : ObjectFile(Binary::ID_COFF, Object), COFFHeader(nullptr),
      COFFBigObjHeader(nullptr), PE32Header(nullptr), PE32PlusHeader(nullptr),
      DataDirectory(nullptr), SectionTable(nullptr), SymbolTable16(nullptr),
      SymbolTable32(nullptr), StringTable(nullptr), StringTableSize(0),
      ImportDirectory(nullptr), NumberOfImportDirectory(0),
      DelayImportDirectory(nullptr), NumberOfDelayImportDirectory(0),
      ExportDirectory(nullptr) {
  // Check that we at least have enough room for a header.
  if (!checkSize(Data, EC, sizeof(coff_file_header)))
    return;

  // The current location in the file where we are looking at.
  uint64_t CurPtr = 0;

  // PE header is optional and is present only in executables. If it exists,
  // it is placed right after COFF header.
  bool HasPEHeader = false;

  // Check if this is a PE/COFF file.
  if (base()[0] == 0x4d && base()[1] == 0x5a) {
    // PE/COFF, seek through MS-DOS compatibility stub and 4-byte
    // PE signature to find 'normal' COFF header.
    if (!checkSize(Data, EC, 0x3c + 8))
      return;
    CurPtr = *reinterpret_cast<const ulittle16_t *>(base() + 0x3c);
    // Check the PE magic bytes. ("PE\0\0")
    if (std::memcmp(base() + CurPtr, COFF::PEMagic, sizeof(COFF::PEMagic)) !=
        0) {
      EC = object_error::parse_failed;
      return;
    }
    CurPtr += sizeof(COFF::PEMagic); // Skip the PE magic bytes.
    HasPEHeader = true;
  }

  if ((EC = getObject(COFFHeader, Data, base() + CurPtr)))
    return;

  // It might be a bigobj file, let's check.  Note that COFF bigobj and COFF
  // import libraries share a common prefix but bigobj is more restrictive.
  if (!HasPEHeader && COFFHeader->Machine == COFF::IMAGE_FILE_MACHINE_UNKNOWN &&
      COFFHeader->NumberOfSections == uint16_t(0xffff) &&
      checkSize(Data, EC, sizeof(coff_bigobj_file_header))) {
    if ((EC = getObject(COFFBigObjHeader, Data, base() + CurPtr)))
      return;

    // Verify that we are dealing with bigobj.
    if (COFFBigObjHeader->Version >= COFF::BigObjHeader::MinBigObjectVersion &&
        std::memcmp(COFFBigObjHeader->UUID, COFF::BigObjMagic,
                    sizeof(COFF::BigObjMagic)) == 0) {
      COFFHeader = nullptr;
      CurPtr += sizeof(coff_bigobj_file_header);
    } else {
      // It's not a bigobj.
      COFFBigObjHeader = nullptr;
    }
  }
  if (COFFHeader) {
    // The prior checkSize call may have failed.  This isn't a hard error
    // because we were just trying to sniff out bigobj.
    EC = object_error::success;
    CurPtr += sizeof(coff_file_header);

    if (COFFHeader->isImportLibrary())
      return;
  }

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

  if ((EC = getObject(SectionTable, Data, base() + CurPtr,
                      getNumberOfSections() * sizeof(coff_section))))
    return;

  // Initialize the pointer to the symbol table.
  if (getPointerToSymbolTable() != 0)
    if ((EC = initSymbolTablePtr()))
      return;

  // Initialize the pointer to the beginning of the import table.
  if ((EC = initImportTablePtr()))
    return;
  if ((EC = initDelayImportTablePtr()))
    return;

  // Initialize the pointer to the export table.
  if ((EC = initExportTablePtr()))
    return;

  EC = object_error::success;
}

basic_symbol_iterator COFFObjectFile::symbol_begin_impl() const {
  DataRefImpl Ret;
  Ret.p = getSymbolTable();
  return basic_symbol_iterator(SymbolRef(Ret, this));
}

basic_symbol_iterator COFFObjectFile::symbol_end_impl() const {
  // The symbol table ends where the string table begins.
  DataRefImpl Ret;
  Ret.p = reinterpret_cast<uintptr_t>(StringTable);
  return basic_symbol_iterator(SymbolRef(Ret, this));
}

import_directory_iterator COFFObjectFile::import_directory_begin() const {
  return import_directory_iterator(
      ImportDirectoryEntryRef(ImportDirectory, 0, this));
}

import_directory_iterator COFFObjectFile::import_directory_end() const {
  return import_directory_iterator(
      ImportDirectoryEntryRef(ImportDirectory, NumberOfImportDirectory, this));
}

delay_import_directory_iterator
COFFObjectFile::delay_import_directory_begin() const {
  return delay_import_directory_iterator(
      DelayImportDirectoryEntryRef(DelayImportDirectory, 0, this));
}

delay_import_directory_iterator
COFFObjectFile::delay_import_directory_end() const {
  return delay_import_directory_iterator(
      DelayImportDirectoryEntryRef(
          DelayImportDirectory, NumberOfDelayImportDirectory, this));
}

export_directory_iterator COFFObjectFile::export_directory_begin() const {
  return export_directory_iterator(
      ExportDirectoryEntryRef(ExportDirectory, 0, this));
}

export_directory_iterator COFFObjectFile::export_directory_end() const {
  if (!ExportDirectory)
    return export_directory_iterator(ExportDirectoryEntryRef(nullptr, 0, this));
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
  int NumSections =
      COFFHeader && COFFHeader->isImportLibrary() ? 0 : getNumberOfSections();
  Ret.p = reinterpret_cast<uintptr_t>(SectionTable + NumSections);
  return section_iterator(SectionRef(Ret, this));
}

uint8_t COFFObjectFile::getBytesInAddress() const {
  return getArch() == Triple::x86_64 ? 8 : 4;
}

StringRef COFFObjectFile::getFileFormatName() const {
  switch(getMachine()) {
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
  switch (getMachine()) {
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

iterator_range<import_directory_iterator>
COFFObjectFile::import_directories() const {
  return make_range(import_directory_begin(), import_directory_end());
}

iterator_range<delay_import_directory_iterator>
COFFObjectFile::delay_import_directories() const {
  return make_range(delay_import_directory_begin(),
                    delay_import_directory_end());
}

iterator_range<export_directory_iterator>
COFFObjectFile::export_directories() const {
  return make_range(export_directory_begin(), export_directory_end());
}

std::error_code COFFObjectFile::getPE32Header(const pe32_header *&Res) const {
  Res = PE32Header;
  return object_error::success;
}

std::error_code
COFFObjectFile::getPE32PlusHeader(const pe32plus_header *&Res) const {
  Res = PE32PlusHeader;
  return object_error::success;
}

std::error_code
COFFObjectFile::getDataDirectory(uint32_t Index,
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

std::error_code COFFObjectFile::getSection(int32_t Index,
                                           const coff_section *&Result) const {
  // Check for special index values.
  if (COFF::isReservedSectionNumber(Index))
    Result = nullptr;
  else if (Index > 0 && static_cast<uint32_t>(Index) <= getNumberOfSections())
    // We already verified the section table data, so no need to check again.
    Result = SectionTable + (Index - 1);
  else
    return object_error::parse_failed;
  return object_error::success;
}

std::error_code COFFObjectFile::getString(uint32_t Offset,
                                          StringRef &Result) const {
  if (StringTableSize <= 4)
    // Tried to get a string from an empty string table.
    return object_error::parse_failed;
  if (Offset >= StringTableSize)
    return object_error::unexpected_eof;
  Result = StringRef(StringTable + Offset);
  return object_error::success;
}

std::error_code COFFObjectFile::getSymbolName(COFFSymbolRef Symbol,
                                              StringRef &Res) const {
  // Check for string table entry. First 4 bytes are 0.
  if (Symbol.getStringTableOffset().Zeroes == 0) {
    uint32_t Offset = Symbol.getStringTableOffset().Offset;
    if (std::error_code EC = getString(Offset, Res))
      return EC;
    return object_error::success;
  }

  if (Symbol.getShortName()[COFF::NameSize - 1] == 0)
    // Null terminated, let ::strlen figure out the length.
    Res = StringRef(Symbol.getShortName());
  else
    // Not null terminated, use all 8 bytes.
    Res = StringRef(Symbol.getShortName(), COFF::NameSize);
  return object_error::success;
}

ArrayRef<uint8_t>
COFFObjectFile::getSymbolAuxData(COFFSymbolRef Symbol) const {
  const uint8_t *Aux = nullptr;

  size_t SymbolSize = getSymbolTableEntrySize();
  if (Symbol.getNumberOfAuxSymbols() > 0) {
    // AUX data comes immediately after the symbol in COFF
    Aux = reinterpret_cast<const uint8_t *>(Symbol.getRawPtr()) + SymbolSize;
# ifndef NDEBUG
    // Verify that the Aux symbol points to a valid entry in the symbol table.
    uintptr_t Offset = uintptr_t(Aux) - uintptr_t(base());
    if (Offset < getPointerToSymbolTable() ||
        Offset >=
            getPointerToSymbolTable() + (getNumberOfSymbols() * SymbolSize))
      report_fatal_error("Aux Symbol data was outside of symbol table.");

    assert((Offset - getPointerToSymbolTable()) % SymbolSize == 0 &&
           "Aux Symbol data did not point to the beginning of a symbol");
# endif
  }
  return makeArrayRef(Aux, Symbol.getNumberOfAuxSymbols() * SymbolSize);
}

std::error_code COFFObjectFile::getSectionName(const coff_section *Sec,
                                               StringRef &Res) const {
  StringRef Name;
  if (Sec->Name[COFF::NameSize - 1] == 0)
    // Null terminated, let ::strlen figure out the length.
    Name = Sec->Name;
  else
    // Not null terminated, use all 8 bytes.
    Name = StringRef(Sec->Name, COFF::NameSize);

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
    if (std::error_code EC = getString(Offset, Name))
      return EC;
  }

  Res = Name;
  return object_error::success;
}

uint64_t COFFObjectFile::getSectionSize(const coff_section *Sec) const {
  // SizeOfRawData and VirtualSize change what they represent depending on
  // whether or not we have an executable image.
  //
  // For object files, SizeOfRawData contains the size of section's data;
  // VirtualSize is always zero.
  //
  // For executables, SizeOfRawData *must* be a multiple of FileAlignment; the
  // actual section size is in VirtualSize.  It is possible for VirtualSize to
  // be greater than SizeOfRawData; the contents past that point should be
  // considered to be zero.
  uint32_t SectionSize;
  if (Sec->VirtualSize)
    SectionSize = std::min(Sec->VirtualSize, Sec->SizeOfRawData);
  else
    SectionSize = Sec->SizeOfRawData;

  return SectionSize;
}

std::error_code
COFFObjectFile::getSectionContents(const coff_section *Sec,
                                   ArrayRef<uint8_t> &Res) const {
  // PointerToRawData and SizeOfRawData won't make sense for BSS sections,
  // don't do anything interesting for them.
  assert((Sec->Characteristics & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA) == 0 &&
         "BSS sections don't have contents!");
  // The only thing that we need to verify is that the contents is contained
  // within the file bounds. We don't need to make sure it doesn't cover other
  // data, as there's nothing that says that is not allowed.
  uintptr_t ConStart = uintptr_t(base()) + Sec->PointerToRawData;
  uint32_t SectionSize = getSectionSize(Sec);
  uintptr_t ConEnd = ConStart + SectionSize;
  if (ConEnd > uintptr_t(Data.getBufferEnd()))
    return object_error::parse_failed;
  Res = makeArrayRef(reinterpret_cast<const uint8_t *>(ConStart), SectionSize);
  return object_error::success;
}

const coff_relocation *COFFObjectFile::toRel(DataRefImpl Rel) const {
  return reinterpret_cast<const coff_relocation*>(Rel.p);
}

void COFFObjectFile::moveRelocationNext(DataRefImpl &Rel) const {
  Rel.p = reinterpret_cast<uintptr_t>(
            reinterpret_cast<const coff_relocation*>(Rel.p) + 1);
}

std::error_code COFFObjectFile::getRelocationAddress(DataRefImpl Rel,
                                                     uint64_t &Res) const {
  report_fatal_error("getRelocationAddress not implemented in COFFObjectFile");
}

std::error_code COFFObjectFile::getRelocationOffset(DataRefImpl Rel,
                                                    uint64_t &Res) const {
  Res = toRel(Rel)->VirtualAddress;
  return object_error::success;
}

symbol_iterator COFFObjectFile::getRelocationSymbol(DataRefImpl Rel) const {
  const coff_relocation *R = toRel(Rel);
  DataRefImpl Ref;
  if (SymbolTable16)
    Ref.p = reinterpret_cast<uintptr_t>(SymbolTable16 + R->SymbolTableIndex);
  else if (SymbolTable32)
    Ref.p = reinterpret_cast<uintptr_t>(SymbolTable32 + R->SymbolTableIndex);
  else
    llvm_unreachable("no symbol table pointer!");
  return symbol_iterator(SymbolRef(Ref, this));
}

std::error_code COFFObjectFile::getRelocationType(DataRefImpl Rel,
                                                  uint64_t &Res) const {
  const coff_relocation* R = toRel(Rel);
  Res = R->Type;
  return object_error::success;
}

const coff_section *
COFFObjectFile::getCOFFSection(const SectionRef &Section) const {
  return toSec(Section.getRawDataRefImpl());
}

COFFSymbolRef COFFObjectFile::getCOFFSymbol(const DataRefImpl &Ref) const {
  if (SymbolTable16)
    return toSymb<coff_symbol16>(Ref);
  if (SymbolTable32)
    return toSymb<coff_symbol32>(Ref);
  llvm_unreachable("no symbol table pointer!");
}

COFFSymbolRef COFFObjectFile::getCOFFSymbol(const SymbolRef &Symbol) const {
  return getCOFFSymbol(Symbol.getRawDataRefImpl());
}

const coff_relocation *
COFFObjectFile::getCOFFRelocation(const RelocationRef &Reloc) const {
  return toRel(Reloc.getRawDataRefImpl());
}

#define LLVM_COFF_SWITCH_RELOC_TYPE_NAME(reloc_type)                           \
  case COFF::reloc_type:                                                       \
    Res = #reloc_type;                                                         \
    break;

std::error_code
COFFObjectFile::getRelocationTypeName(DataRefImpl Rel,
                                      SmallVectorImpl<char> &Result) const {
  const coff_relocation *Reloc = toRel(Rel);
  StringRef Res;
  switch (getMachine()) {
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
  case COFF::IMAGE_FILE_MACHINE_ARMNT:
    switch (Reloc->Type) {
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_ARM_ABSOLUTE);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_ARM_ADDR32);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_ARM_ADDR32NB);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_ARM_BRANCH24);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_ARM_BRANCH11);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_ARM_TOKEN);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_ARM_BLX24);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_ARM_BLX11);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_ARM_SECTION);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_ARM_SECREL);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_ARM_MOV32A);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_ARM_MOV32T);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_ARM_BRANCH20T);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_ARM_BRANCH24T);
    LLVM_COFF_SWITCH_RELOC_TYPE_NAME(IMAGE_REL_ARM_BLX23T);
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

std::error_code
COFFObjectFile::getRelocationValueString(DataRefImpl Rel,
                                         SmallVectorImpl<char> &Result) const {
  const coff_relocation *Reloc = toRel(Rel);
  DataRefImpl Sym;
  ErrorOr<COFFSymbolRef> Symb = getSymbol(Reloc->SymbolTableIndex);
  if (std::error_code EC = Symb.getError())
    return EC;
  Sym.p = reinterpret_cast<uintptr_t>(Symb->getRawPtr());
  StringRef SymName;
  if (std::error_code EC = getSymbolName(Sym, SymName))
    return EC;
  Result.append(SymName.begin(), SymName.end());
  return object_error::success;
}

bool COFFObjectFile::isRelocatableObject() const {
  return !DataDirectory;
}

bool ImportDirectoryEntryRef::
operator==(const ImportDirectoryEntryRef &Other) const {
  return ImportTable == Other.ImportTable && Index == Other.Index;
}

void ImportDirectoryEntryRef::moveNext() {
  ++Index;
}

std::error_code ImportDirectoryEntryRef::getImportTableEntry(
    const import_directory_table_entry *&Result) const {
  Result = ImportTable + Index;
  return object_error::success;
}

static imported_symbol_iterator
makeImportedSymbolIterator(const COFFObjectFile *Object,
                           uintptr_t Ptr, int Index) {
  if (Object->getBytesInAddress() == 4) {
    auto *P = reinterpret_cast<const import_lookup_table_entry32 *>(Ptr);
    return imported_symbol_iterator(ImportedSymbolRef(P, Index, Object));
  }
  auto *P = reinterpret_cast<const import_lookup_table_entry64 *>(Ptr);
  return imported_symbol_iterator(ImportedSymbolRef(P, Index, Object));
}

static imported_symbol_iterator
importedSymbolBegin(uint32_t RVA, const COFFObjectFile *Object) {
  uintptr_t IntPtr = 0;
  Object->getRvaPtr(RVA, IntPtr);
  return makeImportedSymbolIterator(Object, IntPtr, 0);
}

static imported_symbol_iterator
importedSymbolEnd(uint32_t RVA, const COFFObjectFile *Object) {
  uintptr_t IntPtr = 0;
  Object->getRvaPtr(RVA, IntPtr);
  // Forward the pointer to the last entry which is null.
  int Index = 0;
  if (Object->getBytesInAddress() == 4) {
    auto *Entry = reinterpret_cast<ulittle32_t *>(IntPtr);
    while (*Entry++)
      ++Index;
  } else {
    auto *Entry = reinterpret_cast<ulittle64_t *>(IntPtr);
    while (*Entry++)
      ++Index;
  }
  return makeImportedSymbolIterator(Object, IntPtr, Index);
}

imported_symbol_iterator
ImportDirectoryEntryRef::imported_symbol_begin() const {
  return importedSymbolBegin(ImportTable[Index].ImportLookupTableRVA,
                             OwningObject);
}

imported_symbol_iterator
ImportDirectoryEntryRef::imported_symbol_end() const {
  return importedSymbolEnd(ImportTable[Index].ImportLookupTableRVA,
                           OwningObject);
}

iterator_range<imported_symbol_iterator>
ImportDirectoryEntryRef::imported_symbols() const {
  return make_range(imported_symbol_begin(), imported_symbol_end());
}

std::error_code ImportDirectoryEntryRef::getName(StringRef &Result) const {
  uintptr_t IntPtr = 0;
  if (std::error_code EC =
          OwningObject->getRvaPtr(ImportTable[Index].NameRVA, IntPtr))
    return EC;
  Result = StringRef(reinterpret_cast<const char *>(IntPtr));
  return object_error::success;
}

std::error_code
ImportDirectoryEntryRef::getImportLookupTableRVA(uint32_t  &Result) const {
  Result = ImportTable[Index].ImportLookupTableRVA;
  return object_error::success;
}

std::error_code
ImportDirectoryEntryRef::getImportAddressTableRVA(uint32_t &Result) const {
  Result = ImportTable[Index].ImportAddressTableRVA;
  return object_error::success;
}

std::error_code ImportDirectoryEntryRef::getImportLookupEntry(
    const import_lookup_table_entry32 *&Result) const {
  uintptr_t IntPtr = 0;
  uint32_t RVA = ImportTable[Index].ImportLookupTableRVA;
  if (std::error_code EC = OwningObject->getRvaPtr(RVA, IntPtr))
    return EC;
  Result = reinterpret_cast<const import_lookup_table_entry32 *>(IntPtr);
  return object_error::success;
}

bool DelayImportDirectoryEntryRef::
operator==(const DelayImportDirectoryEntryRef &Other) const {
  return Table == Other.Table && Index == Other.Index;
}

void DelayImportDirectoryEntryRef::moveNext() {
  ++Index;
}

imported_symbol_iterator
DelayImportDirectoryEntryRef::imported_symbol_begin() const {
  return importedSymbolBegin(Table[Index].DelayImportNameTable,
                             OwningObject);
}

imported_symbol_iterator
DelayImportDirectoryEntryRef::imported_symbol_end() const {
  return importedSymbolEnd(Table[Index].DelayImportNameTable,
                           OwningObject);
}

iterator_range<imported_symbol_iterator>
DelayImportDirectoryEntryRef::imported_symbols() const {
  return make_range(imported_symbol_begin(), imported_symbol_end());
}

std::error_code DelayImportDirectoryEntryRef::getName(StringRef &Result) const {
  uintptr_t IntPtr = 0;
  if (std::error_code EC = OwningObject->getRvaPtr(Table[Index].Name, IntPtr))
    return EC;
  Result = StringRef(reinterpret_cast<const char *>(IntPtr));
  return object_error::success;
}

std::error_code DelayImportDirectoryEntryRef::
getDelayImportTable(const delay_import_directory_table_entry *&Result) const {
  Result = Table;
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
std::error_code ExportDirectoryEntryRef::getDllName(StringRef &Result) const {
  uintptr_t IntPtr = 0;
  if (std::error_code EC =
          OwningObject->getRvaPtr(ExportTable->NameRVA, IntPtr))
    return EC;
  Result = StringRef(reinterpret_cast<const char *>(IntPtr));
  return object_error::success;
}

// Returns the starting ordinal number.
std::error_code
ExportDirectoryEntryRef::getOrdinalBase(uint32_t &Result) const {
  Result = ExportTable->OrdinalBase;
  return object_error::success;
}

// Returns the export ordinal of the current export symbol.
std::error_code ExportDirectoryEntryRef::getOrdinal(uint32_t &Result) const {
  Result = ExportTable->OrdinalBase + Index;
  return object_error::success;
}

// Returns the address of the current export symbol.
std::error_code ExportDirectoryEntryRef::getExportRVA(uint32_t &Result) const {
  uintptr_t IntPtr = 0;
  if (std::error_code EC =
          OwningObject->getRvaPtr(ExportTable->ExportAddressTableRVA, IntPtr))
    return EC;
  const export_address_table_entry *entry =
      reinterpret_cast<const export_address_table_entry *>(IntPtr);
  Result = entry[Index].ExportRVA;
  return object_error::success;
}

// Returns the name of the current export symbol. If the symbol is exported only
// by ordinal, the empty string is set as a result.
std::error_code
ExportDirectoryEntryRef::getSymbolName(StringRef &Result) const {
  uintptr_t IntPtr = 0;
  if (std::error_code EC =
          OwningObject->getRvaPtr(ExportTable->OrdinalTableRVA, IntPtr))
    return EC;
  const ulittle16_t *Start = reinterpret_cast<const ulittle16_t *>(IntPtr);

  uint32_t NumEntries = ExportTable->NumberOfNamePointers;
  int Offset = 0;
  for (const ulittle16_t *I = Start, *E = Start + NumEntries;
       I < E; ++I, ++Offset) {
    if (*I != Index)
      continue;
    if (std::error_code EC =
            OwningObject->getRvaPtr(ExportTable->NamePointerRVA, IntPtr))
      return EC;
    const ulittle32_t *NamePtr = reinterpret_cast<const ulittle32_t *>(IntPtr);
    if (std::error_code EC = OwningObject->getRvaPtr(NamePtr[Offset], IntPtr))
      return EC;
    Result = StringRef(reinterpret_cast<const char *>(IntPtr));
    return object_error::success;
  }
  Result = "";
  return object_error::success;
}

bool ImportedSymbolRef::
operator==(const ImportedSymbolRef &Other) const {
  return Entry32 == Other.Entry32 && Entry64 == Other.Entry64
      && Index == Other.Index;
}

void ImportedSymbolRef::moveNext() {
  ++Index;
}

std::error_code
ImportedSymbolRef::getSymbolName(StringRef &Result) const {
  uint32_t RVA;
  if (Entry32) {
    // If a symbol is imported only by ordinal, it has no name.
    if (Entry32[Index].isOrdinal())
      return object_error::success;
    RVA = Entry32[Index].getHintNameRVA();
  } else {
    if (Entry64[Index].isOrdinal())
      return object_error::success;
    RVA = Entry64[Index].getHintNameRVA();
  }
  uintptr_t IntPtr = 0;
  if (std::error_code EC = OwningObject->getRvaPtr(RVA, IntPtr))
    return EC;
  // +2 because the first two bytes is hint.
  Result = StringRef(reinterpret_cast<const char *>(IntPtr + 2));
  return object_error::success;
}

std::error_code ImportedSymbolRef::getOrdinal(uint16_t &Result) const {
  uint32_t RVA;
  if (Entry32) {
    if (Entry32[Index].isOrdinal()) {
      Result = Entry32[Index].getOrdinal();
      return object_error::success;
    }
    RVA = Entry32[Index].getHintNameRVA();
  } else {
    if (Entry64[Index].isOrdinal()) {
      Result = Entry64[Index].getOrdinal();
      return object_error::success;
    }
    RVA = Entry64[Index].getHintNameRVA();
  }
  uintptr_t IntPtr = 0;
  if (std::error_code EC = OwningObject->getRvaPtr(RVA, IntPtr))
    return EC;
  Result = *reinterpret_cast<const ulittle16_t *>(IntPtr);
  return object_error::success;
}

ErrorOr<std::unique_ptr<COFFObjectFile>>
ObjectFile::createCOFFObjectFile(MemoryBufferRef Object) {
  std::error_code EC;
  std::unique_ptr<COFFObjectFile> Ret(new COFFObjectFile(Object, EC));
  if (EC)
    return EC;
  return std::move(Ret);
}
