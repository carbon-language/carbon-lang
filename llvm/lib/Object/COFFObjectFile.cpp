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

namespace {
// Returns false if size is greater than the buffer size. And sets ec.
bool checkSize(const MemoryBuffer *m, error_code &ec, uint64_t size) {
  if (m->getBufferSize() < size) {
    ec = object_error::unexpected_eof;
    return false;
  }
  return true;
}

// Returns false if any bytes in [addr, addr + size) fall outsize of m.
bool checkAddr(const MemoryBuffer *m,
               error_code &ec,
               uintptr_t addr,
               uint64_t size) {
  if (addr + size < addr ||
      addr + size < size ||
      addr + size > uintptr_t(m->getBufferEnd())) {
    ec = object_error::unexpected_eof;
    return false;
  }
  return true;
}
}

const coff_symbol *COFFObjectFile::toSymb(DataRefImpl Symb) const {
  const coff_symbol *addr = reinterpret_cast<const coff_symbol*>(Symb.p);

# ifndef NDEBUG
  // Verify that the symbol points to a valid entry in the symbol table.
  uintptr_t offset = uintptr_t(addr) - uintptr_t(base());
  if (offset < Header->PointerToSymbolTable
      || offset >= Header->PointerToSymbolTable
         + (Header->NumberOfSymbols * sizeof(coff_symbol)))
    report_fatal_error("Symbol was outside of symbol table.");

  assert((offset - Header->PointerToSymbolTable) % sizeof(coff_symbol)
         == 0 && "Symbol did not point to the beginning of a symbol");
# endif

  return addr;
}

const coff_section *COFFObjectFile::toSec(DataRefImpl Sec) const {
  const coff_section *addr = reinterpret_cast<const coff_section*>(Sec.p);

# ifndef NDEBUG
  // Verify that the section points to a valid entry in the section table.
  if (addr < SectionTable
      || addr >= (SectionTable + Header->NumberOfSections))
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
  // Check for string table entry. First 4 bytes are 0.
  if (symb->Name.Offset.Zeroes == 0) {
    uint32_t Offset = symb->Name.Offset.Offset;
    if (error_code ec = getString(Offset, Result))
      return ec;
    return object_error::success;
  }

  if (symb->Name.ShortName[7] == 0)
    // Null terminated, let ::strlen figure out the length.
    Result = StringRef(symb->Name.ShortName);
  else
    // Not null terminated, use all 8 bytes.
    Result = StringRef(symb->Name.ShortName, 8);
  return object_error::success;
}

error_code COFFObjectFile::getSymbolOffset(DataRefImpl Symb,
                                            uint64_t &Result) const {
  const coff_symbol *symb = toSymb(Symb);
  const coff_section *Section = NULL;
  if (error_code ec = getSection(symb->SectionNumber, Section))
    return ec;
  char Type;
  if (error_code ec = getSymbolNMTypeChar(Symb, Type))
    return ec;
  if (Type == 'U' || Type == 'w')
    Result = UnknownAddressOrSize;
  else if (Section)
    Result = Section->VirtualAddress + symb->Value;
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
  char Type;
  if (error_code ec = getSymbolNMTypeChar(Symb, Type))
    return ec;
  if (Type == 'U' || Type == 'w')
    Result = UnknownAddressOrSize;
  else if (Section)
    Result = reinterpret_cast<uintptr_t>(base() +
                                         Section->PointerToRawData +
                                         symb->Value);
  else
    Result = reinterpret_cast<uintptr_t>(base() + symb->Value);
  return object_error::success;
}

error_code COFFObjectFile::getSymbolType(DataRefImpl Symb,
                                         SymbolRef::SymbolType &Result) const {
  const coff_symbol *symb = toSymb(Symb);
  Result = SymbolRef::ST_Other;
  if (symb->StorageClass == COFF::IMAGE_SYM_CLASS_EXTERNAL &&
      symb->SectionNumber == COFF::IMAGE_SYM_UNDEFINED) {
    Result = SymbolRef::ST_External;
  } else {
    if (symb->Type.ComplexType == COFF::IMAGE_SYM_DTYPE_FUNCTION) {
      Result = SymbolRef::ST_Function;
    } else {
      char Type;
      if (error_code ec = getSymbolNMTypeChar(Symb, Type))
        return ec;
      if (Type == 'r' || Type == 'R') {
        Result = SymbolRef::ST_Data;
      }
    }
  }
  return object_error::success;
}

error_code COFFObjectFile::isSymbolGlobal(DataRefImpl Symb,
                                          bool &Result) const {
  const coff_symbol *symb = toSymb(Symb);
  Result = (symb->StorageClass == COFF::IMAGE_SYM_CLASS_EXTERNAL);
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
  char Type;
  if (error_code ec = getSymbolNMTypeChar(Symb, Type))
    return ec;
  if (Type == 'U' || Type == 'w')
    Result = UnknownAddressOrSize;
  else if (Section)
    Result = Section->SizeOfRawData - symb->Value;
  else
    Result = 0;
  return object_error::success;
}

error_code COFFObjectFile::getSymbolNMTypeChar(DataRefImpl Symb,
                                               char &Result) const {
  const coff_symbol *symb = toSymb(Symb);
  StringRef name;
  if (error_code ec = getSymbolName(Symb, name))
    return ec;
  char ret = StringSwitch<char>(name)
    .StartsWith(".debug", 'N')
    .StartsWith(".sxdata", 'N')
    .Default('?');

  if (ret != '?') {
    Result = ret;
    return object_error::success;
  }

  uint32_t Characteristics = 0;
  if (symb->SectionNumber > 0) {
    const coff_section *Section = NULL;
    if (error_code ec = getSection(symb->SectionNumber, Section))
      return ec;
    Characteristics = Section->Characteristics;
  }

  switch (symb->SectionNumber) {
  case COFF::IMAGE_SYM_UNDEFINED:
    // Check storage classes.
    if (symb->StorageClass == COFF::IMAGE_SYM_CLASS_WEAK_EXTERNAL) {
      Result = 'w';
      return object_error::success; // Don't do ::toupper.
    } else
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

  Result = ret;
  return object_error::success;
}

error_code COFFObjectFile::isSymbolInternal(DataRefImpl Symb,
                                            bool &Result) const {
  Result = false;
  return object_error::success;
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
    if (error_code ec = getString(Offset, name))
      return ec;
  }

  Result = name;
  return object_error::success;
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
  // The only thing that we need to verify is that the contents is contained
  // within the file bounds. We don't need to make sure it doesn't cover other
  // data, as there's nothing that says that is not allowed.
  uintptr_t con_start = uintptr_t(base()) + sec->PointerToRawData;
  uintptr_t con_end = con_start + sec->SizeOfRawData;
  if (con_end >= uintptr_t(Data->getBufferEnd()))
    return object_error::parse_failed;
  Result = StringRef(reinterpret_cast<const char*>(con_start),
                     sec->SizeOfRawData);
  return object_error::success;
}

error_code COFFObjectFile::isSectionText(DataRefImpl Sec,
                                         bool &Result) const {
  const coff_section *sec = toSec(Sec);
  Result = sec->Characteristics & COFF::IMAGE_SCN_CNT_CODE;
  return object_error::success;
}

error_code COFFObjectFile::sectionContainsSymbol(DataRefImpl Sec,
                                                 DataRefImpl Symb,
                                                 bool &Result) const {
  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

COFFObjectFile::COFFObjectFile(MemoryBuffer *Object, error_code &ec)
  : ObjectFile(Binary::isCOFF, Object, ec) {
  // Check that we at least have enough room for a header.
  if (!checkSize(Data, ec, sizeof(coff_file_header))) return;

  // The actual starting location of the COFF header in the file. This can be
  // non-zero in PE/COFF files.
  uint64_t HeaderStart = 0;

  // Check if this is a PE/COFF file.
  if (base()[0] == 0x4d && base()[1] == 0x5a) {
    // PE/COFF, seek through MS-DOS compatibility stub and 4-byte
    // PE signature to find 'normal' COFF header.
    if (!checkSize(Data, ec, 0x3c + 8)) return;
    HeaderStart += *reinterpret_cast<const ulittle32_t *>(base() + 0x3c);
    // Check the PE header. ("PE\0\0")
    if (std::memcmp(base() + HeaderStart, "PE\0\0", 4) != 0) {
      ec = object_error::parse_failed;
      return;
    }
    HeaderStart += 4; // Skip the PE Header.
  }

  Header = reinterpret_cast<const coff_file_header *>(base() + HeaderStart);
  if (!checkAddr(Data, ec, uintptr_t(Header), sizeof(coff_file_header)))
    return;

  SectionTable =
    reinterpret_cast<const coff_section *>( base()
                                          + HeaderStart
                                          + sizeof(coff_file_header)
                                          + Header->SizeOfOptionalHeader);
  if (!checkAddr(Data, ec, uintptr_t(SectionTable),
                 Header->NumberOfSections * sizeof(coff_section)))
    return;

  SymbolTable =
    reinterpret_cast<const coff_symbol *>(base()
                                          + Header->PointerToSymbolTable);
  if (!checkAddr(Data, ec, uintptr_t(SymbolTable),
                 Header->NumberOfSymbols * sizeof(coff_symbol)))
    return;

  // Find string table.
  StringTable = reinterpret_cast<const char *>(base())
                + Header->PointerToSymbolTable
                + Header->NumberOfSymbols * sizeof(coff_symbol);
  if (!checkAddr(Data, ec, uintptr_t(StringTable), sizeof(ulittle32_t)))
    return;

  StringTableSize = *reinterpret_cast<const ulittle32_t *>(StringTable);
  if (!checkAddr(Data, ec, uintptr_t(StringTable), StringTableSize))
    return;
  // Check that the string table is null terminated if has any in it.
  if (StringTableSize < 4
      || (StringTableSize > 4 && StringTable[StringTableSize - 1] != 0)) {
    ec = object_error::parse_failed;
    return;
  }

  ec = object_error::success;
}

ObjectFile::symbol_iterator COFFObjectFile::begin_symbols() const {
  DataRefImpl ret;
  std::memset(&ret, 0, sizeof(DataRefImpl));
  ret.p = reinterpret_cast<intptr_t>(SymbolTable);
  return symbol_iterator(SymbolRef(ret, this));
}

ObjectFile::symbol_iterator COFFObjectFile::end_symbols() const {
  // The symbol table ends where the string table begins.
  DataRefImpl ret;
  std::memset(&ret, 0, sizeof(DataRefImpl));
  ret.p = reinterpret_cast<intptr_t>(StringTable);
  return symbol_iterator(SymbolRef(ret, this));
}

ObjectFile::section_iterator COFFObjectFile::begin_sections() const {
  DataRefImpl ret;
  std::memset(&ret, 0, sizeof(DataRefImpl));
  ret.p = reinterpret_cast<intptr_t>(SectionTable);
  return section_iterator(SectionRef(ret, this));
}

ObjectFile::section_iterator COFFObjectFile::end_sections() const {
  DataRefImpl ret;
  std::memset(&ret, 0, sizeof(DataRefImpl));
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

error_code COFFObjectFile::getSection(int32_t index,
                                      const coff_section *&Result) const {
  // Check for special index values.
  if (index == COFF::IMAGE_SYM_UNDEFINED ||
      index == COFF::IMAGE_SYM_ABSOLUTE ||
      index == COFF::IMAGE_SYM_DEBUG)
    Result = NULL;
  else if (index > 0 && index <= Header->NumberOfSections)
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

const coff_relocation *COFFObjectFile::toRel(DataRefImpl Rel) const {
  assert(Rel.d.b < Header->NumberOfSections && "Section index out of range!");
  const coff_section *Sect = NULL;
  getSection(Rel.d.b, Sect);
  assert(Rel.d.a < Sect->NumberOfRelocations && "Relocation index out of range!");
  return
    reinterpret_cast<const coff_relocation*>(base() +
                                             Sect->PointerToRelocations) +
                                             Rel.d.a;
}
error_code COFFObjectFile::getRelocationNext(DataRefImpl Rel,
                                             RelocationRef &Res) const {
  const coff_section *Sect = NULL;
  if (error_code ec = getSection(Rel.d.b, Sect))
    return ec;
  if (++Rel.d.a >= Sect->NumberOfRelocations) {
    Rel.d.a = 0;
    while (++Rel.d.b < Header->NumberOfSections) {
      const coff_section *Sect = NULL;
      getSection(Rel.d.b, Sect);
      if (Sect->NumberOfRelocations > 0)
        break;
    }
  }
  Res = RelocationRef(Rel, this);
  return object_error::success;
}
error_code COFFObjectFile::getRelocationAddress(DataRefImpl Rel,
                                                uint64_t &Res) const {
  const coff_section *Sect = NULL;
  if (error_code ec = getSection(Rel.d.b, Sect))
    return ec;
  const coff_relocation* R = toRel(Rel);
  Res = reinterpret_cast<uintptr_t>(base() +
                                    Sect->PointerToRawData +
                                    R->VirtualAddress);
  return object_error::success;
}
error_code COFFObjectFile::getRelocationSymbol(DataRefImpl Rel,
                                               SymbolRef &Res) const {
  const coff_relocation* R = toRel(Rel);
  DataRefImpl Symb;
  Symb.p = reinterpret_cast<uintptr_t>(SymbolTable + R->SymbolTableIndex);
  Res = SymbolRef(Symb, this);
  return object_error::success;
}
error_code COFFObjectFile::getRelocationType(DataRefImpl Rel,
                                             uint32_t &Res) const {
  const coff_relocation* R = toRel(Rel);
  Res = R->Type;
  return object_error::success;
}
error_code COFFObjectFile::getRelocationAdditionalInfo(DataRefImpl Rel,
                                                       int64_t &Res) const {
  Res = 0;
  return object_error::success;
}
ObjectFile::relocation_iterator COFFObjectFile::begin_relocations() const {
  DataRefImpl ret;
  ret.d.a = 0;
  ret.d.b = 1;
  return relocation_iterator(RelocationRef(ret, this));
}
ObjectFile::relocation_iterator COFFObjectFile::end_relocations() const {
  DataRefImpl ret;
  ret.d.a = 0;
  ret.d.b = Header->NumberOfSections;
  return relocation_iterator(RelocationRef(ret, this));
}


namespace llvm {

  ObjectFile *ObjectFile::createCOFFObjectFile(MemoryBuffer *Object) {
    error_code ec;
    return new COFFObjectFile(Object, ec);
  }

} // end namespace llvm
