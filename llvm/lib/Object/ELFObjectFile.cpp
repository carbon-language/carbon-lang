//===- ELFObjectFile.cpp - ELF object file implementation -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ELFObjectFile class.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include <limits>
#include <utility>

using namespace llvm;
using namespace object;

// Templates to choose Elf_Addr and Elf_Off depending on is64Bits.
namespace {
template<support::endianness target_endianness>
struct ELFDataTypeTypedefHelperCommon {
  typedef support::detail::packed_endian_specific_integral
    <uint16_t, target_endianness, support::aligned> Elf_Half;
  typedef support::detail::packed_endian_specific_integral
    <uint32_t, target_endianness, support::aligned> Elf_Word;
  typedef support::detail::packed_endian_specific_integral
    <int32_t, target_endianness, support::aligned> Elf_Sword;
  typedef support::detail::packed_endian_specific_integral
    <uint64_t, target_endianness, support::aligned> Elf_Xword;
  typedef support::detail::packed_endian_specific_integral
    <int64_t, target_endianness, support::aligned> Elf_Sxword;
};
}

namespace {
template<support::endianness target_endianness, bool is64Bits>
struct ELFDataTypeTypedefHelper;

/// ELF 32bit types.
template<support::endianness target_endianness>
struct ELFDataTypeTypedefHelper<target_endianness, false>
  : ELFDataTypeTypedefHelperCommon<target_endianness> {
  typedef support::detail::packed_endian_specific_integral
    <uint32_t, target_endianness, support::aligned> Elf_Addr;
  typedef support::detail::packed_endian_specific_integral
    <uint32_t, target_endianness, support::aligned> Elf_Off;
};

/// ELF 64bit types.
template<support::endianness target_endianness>
struct ELFDataTypeTypedefHelper<target_endianness, true>
  : ELFDataTypeTypedefHelperCommon<target_endianness>{
  typedef support::detail::packed_endian_specific_integral
    <uint64_t, target_endianness, support::aligned> Elf_Addr;
  typedef support::detail::packed_endian_specific_integral
    <uint64_t, target_endianness, support::aligned> Elf_Off;
};
}

// I really don't like doing this, but the alternative is copypasta.
#define LLVM_ELF_IMPORT_TYPES(target_endianness, is64Bits) \
typedef typename \
  ELFDataTypeTypedefHelper<target_endianness, is64Bits>::Elf_Addr Elf_Addr; \
typedef typename \
  ELFDataTypeTypedefHelper<target_endianness, is64Bits>::Elf_Off Elf_Off; \
typedef typename \
  ELFDataTypeTypedefHelper<target_endianness, is64Bits>::Elf_Half Elf_Half; \
typedef typename \
  ELFDataTypeTypedefHelper<target_endianness, is64Bits>::Elf_Word Elf_Word; \
typedef typename \
  ELFDataTypeTypedefHelper<target_endianness, is64Bits>::Elf_Sword Elf_Sword; \
typedef typename \
  ELFDataTypeTypedefHelper<target_endianness, is64Bits>::Elf_Xword Elf_Xword; \
typedef typename \
  ELFDataTypeTypedefHelper<target_endianness, is64Bits>::Elf_Sxword Elf_Sxword;

  // Section header.
namespace {
template<support::endianness target_endianness, bool is64Bits>
struct Elf_Shdr_Base;

template<support::endianness target_endianness>
struct Elf_Shdr_Base<target_endianness, false> {
  LLVM_ELF_IMPORT_TYPES(target_endianness, false)
  Elf_Word sh_name;     // Section name (index into string table)
  Elf_Word sh_type;     // Section type (SHT_*)
  Elf_Word sh_flags;    // Section flags (SHF_*)
  Elf_Addr sh_addr;     // Address where section is to be loaded
  Elf_Off  sh_offset;   // File offset of section data, in bytes
  Elf_Word sh_size;     // Size of section, in bytes
  Elf_Word sh_link;     // Section type-specific header table index link
  Elf_Word sh_info;     // Section type-specific extra information
  Elf_Word sh_addralign;// Section address alignment
  Elf_Word sh_entsize;  // Size of records contained within the section
};

template<support::endianness target_endianness>
struct Elf_Shdr_Base<target_endianness, true> {
  LLVM_ELF_IMPORT_TYPES(target_endianness, true)
  Elf_Word  sh_name;     // Section name (index into string table)
  Elf_Word  sh_type;     // Section type (SHT_*)
  Elf_Xword sh_flags;    // Section flags (SHF_*)
  Elf_Addr  sh_addr;     // Address where section is to be loaded
  Elf_Off   sh_offset;   // File offset of section data, in bytes
  Elf_Xword sh_size;     // Size of section, in bytes
  Elf_Word  sh_link;     // Section type-specific header table index link
  Elf_Word  sh_info;     // Section type-specific extra information
  Elf_Xword sh_addralign;// Section address alignment
  Elf_Xword sh_entsize;  // Size of records contained within the section
};

template<support::endianness target_endianness, bool is64Bits>
struct Elf_Shdr_Impl : Elf_Shdr_Base<target_endianness, is64Bits> {
  using Elf_Shdr_Base<target_endianness, is64Bits>::sh_entsize;
  using Elf_Shdr_Base<target_endianness, is64Bits>::sh_size;

  /// @brief Get the number of entities this section contains if it has any.
  unsigned getEntityCount() const {
    if (sh_entsize == 0)
      return 0;
    else
      return sh_size / sh_entsize;
  }
};
}

namespace {
template<support::endianness target_endianness, bool is64Bits>
struct Elf_Sym_Base;

template<support::endianness target_endianness>
struct Elf_Sym_Base<target_endianness, false> {
  LLVM_ELF_IMPORT_TYPES(target_endianness, false)
  Elf_Word      st_name;  // Symbol name (index into string table)
  Elf_Addr      st_value; // Value or address associated with the symbol
  Elf_Word      st_size;  // Size of the symbol
  unsigned char st_info;  // Symbol's type and binding attributes
  unsigned char st_other; // Must be zero; reserved
  Elf_Half      st_shndx; // Which section (header table index) it's defined in
};

template<support::endianness target_endianness>
struct Elf_Sym_Base<target_endianness, true> {
  LLVM_ELF_IMPORT_TYPES(target_endianness, true)
  Elf_Word      st_name;  // Symbol name (index into string table)
  unsigned char st_info;  // Symbol's type and binding attributes
  unsigned char st_other; // Must be zero; reserved
  Elf_Half      st_shndx; // Which section (header table index) it's defined in
  Elf_Addr      st_value; // Value or address associated with the symbol
  Elf_Xword     st_size;  // Size of the symbol
};

template<support::endianness target_endianness, bool is64Bits>
struct Elf_Sym_Impl : Elf_Sym_Base<target_endianness, is64Bits> {
  using Elf_Sym_Base<target_endianness, is64Bits>::st_info;

  // These accessors and mutators correspond to the ELF32_ST_BIND,
  // ELF32_ST_TYPE, and ELF32_ST_INFO macros defined in the ELF specification:
  unsigned char getBinding() const { return st_info >> 4; }
  unsigned char getType() const { return st_info & 0x0f; }
  void setBinding(unsigned char b) { setBindingAndType(b, getType()); }
  void setType(unsigned char t) { setBindingAndType(getBinding(), t); }
  void setBindingAndType(unsigned char b, unsigned char t) {
    st_info = (b << 4) + (t & 0x0f);
  }
};
}

namespace {
struct ELFDataRefImpl {
  uint32_t SymbolIndex;
  uint16_t SymbolTableSectionIndex;
  uint16_t Unused;
};
}

namespace {
template<support::endianness target_endianness, bool is64Bits>
class ELFObjectFile : public ObjectFile {
  LLVM_ELF_IMPORT_TYPES(target_endianness, is64Bits)

  typedef Elf_Shdr_Impl<target_endianness, is64Bits> Elf_Shdr;
  typedef Elf_Sym_Impl<target_endianness, is64Bits> Elf_Sym;

  struct Elf_Ehdr {
    unsigned char e_ident[ELF::EI_NIDENT]; // ELF Identification bytes
    Elf_Half e_type;     // Type of file (see ET_*)
    Elf_Half e_machine;  // Required architecture for this file (see EM_*)
    Elf_Word e_version;  // Must be equal to 1
    Elf_Addr e_entry;    // Address to jump to in order to start program
    Elf_Off  e_phoff;    // Program header table's file offset, in bytes
    Elf_Off  e_shoff;    // Section header table's file offset, in bytes
    Elf_Word e_flags;    // Processor-specific flags
    Elf_Half e_ehsize;   // Size of ELF header, in bytes
    Elf_Half e_phentsize;// Size of an entry in the program header table
    Elf_Half e_phnum;    // Number of entries in the program header table
    Elf_Half e_shentsize;// Size of an entry in the section header table
    Elf_Half e_shnum;    // Number of entries in the section header table
    Elf_Half e_shstrndx; // Section header table index of section name
                                  // string table
    bool checkMagic() const {
      return (memcmp(e_ident, ELF::ElfMagic, strlen(ELF::ElfMagic))) == 0;
    }
    unsigned char getFileClass() const { return e_ident[ELF::EI_CLASS]; }
    unsigned char getDataEncoding() const { return e_ident[ELF::EI_DATA]; }
  };

  typedef SmallVector<const Elf_Shdr*, 1> SymbolTableSections_t;

  const Elf_Ehdr *Header;
  const Elf_Shdr *SectionHeaderTable;
  const Elf_Shdr *dot_shstrtab_sec; // Section header string table.
  const Elf_Shdr *dot_strtab_sec;   // Symbol header string table.
  SymbolTableSections_t SymbolTableSections;

  void            validateSymbol(DataRefImpl Symb) const;
  const Elf_Sym  *getSymbol(DataRefImpl Symb) const;
  const Elf_Shdr *getSection(DataRefImpl index) const;
  const Elf_Shdr *getSection(uint16_t index) const;
  const char     *getString(uint16_t section, uint32_t offset) const;
  const char     *getString(const Elf_Shdr *section, uint32_t offset) const;

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
  ELFObjectFile(MemoryBuffer *Object);
  virtual symbol_iterator begin_symbols() const;
  virtual symbol_iterator end_symbols() const;
  virtual section_iterator begin_sections() const;
  virtual section_iterator end_sections() const;

  virtual uint8_t getBytesInAddress() const;
  virtual StringRef getFileFormatName() const;
  virtual unsigned getArch() const;
};
} // end namespace

template<support::endianness target_endianness, bool is64Bits>
void ELFObjectFile<target_endianness, is64Bits>
                  ::validateSymbol(DataRefImpl Symb) const {
  const ELFDataRefImpl SymbolData = *reinterpret_cast<ELFDataRefImpl *>(&Symb);
  const Elf_Sym  *symb = getSymbol(Symb);
  const Elf_Shdr *SymbolTableSection =
    SymbolTableSections[SymbolData.SymbolTableSectionIndex];
  // FIXME: We really need to do proper error handling in the case of an invalid
  //        input file. Because we don't use exceptions, I think we'll just pass
  //        an error object around.
  if (!(  symb
        && SymbolTableSection
        && symb >= (const Elf_Sym*)(base
                   + SymbolTableSection->sh_offset)
        && symb <  (const Elf_Sym*)(base
                   + SymbolTableSection->sh_offset
                   + SymbolTableSection->sh_size)))
    // FIXME: Proper error handling.
    report_fatal_error("Symb must point to a valid symbol!");
}

template<support::endianness target_endianness, bool is64Bits>
SymbolRef ELFObjectFile<target_endianness, is64Bits>
                       ::getSymbolNext(DataRefImpl Symb) const {
  validateSymbol(Symb);
  ELFDataRefImpl &SymbolData = *reinterpret_cast<ELFDataRefImpl *>(&Symb);
  const Elf_Shdr *SymbolTableSection =
    SymbolTableSections[SymbolData.SymbolTableSectionIndex];

  ++SymbolData.SymbolIndex;
  // Check to see if we are at the end of this symbol table.
  if (SymbolData.SymbolIndex >= SymbolTableSection->getEntityCount()) {
    // We are at the end. If there are other symbol tables, jump to them.
    ++SymbolData.SymbolTableSectionIndex;
    SymbolData.SymbolIndex = 1; // The 0th symbol in ELF is fake.
    // Otherwise return the terminator.
    if (SymbolData.SymbolTableSectionIndex >= SymbolTableSections.size()) {
      SymbolData.SymbolIndex = std::numeric_limits<uint32_t>::max();
      SymbolData.SymbolTableSectionIndex = std::numeric_limits<uint32_t>::max();
    }
  }

  return SymbolRef(Symb, this);
}

template<support::endianness target_endianness, bool is64Bits>
StringRef ELFObjectFile<target_endianness, is64Bits>
                       ::getSymbolName(DataRefImpl Symb) const {
  validateSymbol(Symb);
  const Elf_Sym  *symb = getSymbol(Symb);
  if (symb->st_name == 0) {
    const Elf_Shdr *section = getSection(symb->st_shndx);
    if (!section)
      return "";
    return getString(dot_shstrtab_sec, section->sh_name);
  }

  // Use the default symbol table name section.
  return getString(dot_strtab_sec, symb->st_name);
}

template<support::endianness target_endianness, bool is64Bits>
uint64_t ELFObjectFile<target_endianness, is64Bits>
                      ::getSymbolAddress(DataRefImpl Symb) const {
  validateSymbol(Symb);
  const Elf_Sym  *symb = getSymbol(Symb);
  const Elf_Shdr *Section;
  switch (symb->st_shndx) {
  case ELF::SHN_COMMON:
   // Undefined symbols have no address yet.
  case ELF::SHN_UNDEF: return UnknownAddressOrSize;
  case ELF::SHN_ABS: return symb->st_value;
  default: Section = getSection(symb->st_shndx);
  }

  switch (symb->getType()) {
  case ELF::STT_SECTION: return Section ? Section->sh_addr
                                        : UnknownAddressOrSize;
  case ELF::STT_FUNC:
  case ELF::STT_OBJECT:
  case ELF::STT_NOTYPE:
    return symb->st_value;
  default: return UnknownAddressOrSize;
  }
}

template<support::endianness target_endianness, bool is64Bits>
uint64_t ELFObjectFile<target_endianness, is64Bits>
                      ::getSymbolSize(DataRefImpl Symb) const {
  validateSymbol(Symb);
  const Elf_Sym  *symb = getSymbol(Symb);
  if (symb->st_size == 0)
    return UnknownAddressOrSize;
  return symb->st_size;
}

template<support::endianness target_endianness, bool is64Bits>
char ELFObjectFile<target_endianness, is64Bits>
                  ::getSymbolNMTypeChar(DataRefImpl Symb) const {
  validateSymbol(Symb);
  const Elf_Sym  *symb = getSymbol(Symb);
  const Elf_Shdr *Section = getSection(symb->st_shndx);

  char ret = '?';

  if (Section) {
    switch (Section->sh_type) {
    case ELF::SHT_PROGBITS:
    case ELF::SHT_DYNAMIC:
      switch (Section->sh_flags) {
      case (ELF::SHF_ALLOC | ELF::SHF_EXECINSTR):
        ret = 't'; break;
      case (ELF::SHF_ALLOC | ELF::SHF_WRITE):
        ret = 'd'; break;
      case ELF::SHF_ALLOC:
      case (ELF::SHF_ALLOC | ELF::SHF_MERGE):
      case (ELF::SHF_ALLOC | ELF::SHF_MERGE | ELF::SHF_STRINGS):
        ret = 'r'; break;
      }
      break;
    case ELF::SHT_NOBITS: ret = 'b';
    }
  }

  switch (symb->st_shndx) {
  case ELF::SHN_UNDEF:
    if (ret == '?')
      ret = 'U';
    break;
  case ELF::SHN_ABS: ret = 'a'; break;
  case ELF::SHN_COMMON: ret = 'c'; break;
  }

  switch (symb->getBinding()) {
  case ELF::STB_GLOBAL: ret = ::toupper(ret); break;
  case ELF::STB_WEAK:
    if (symb->st_shndx == ELF::SHN_UNDEF)
      ret = 'w';
    else
      if (symb->getType() == ELF::STT_OBJECT)
        ret = 'V';
      else
        ret = 'W';
  }

  if (ret == '?' && symb->getType() == ELF::STT_SECTION)
    return StringSwitch<char>(getSymbolName(Symb))
      .StartsWith(".debug", 'N')
      .StartsWith(".note", 'n');

  return ret;
}

template<support::endianness target_endianness, bool is64Bits>
bool ELFObjectFile<target_endianness, is64Bits>
                  ::isSymbolInternal(DataRefImpl Symb) const {
  validateSymbol(Symb);
  const Elf_Sym  *symb = getSymbol(Symb);

  if (  symb->getType() == ELF::STT_FILE
     || symb->getType() == ELF::STT_SECTION)
    return true;
  return false;
}

template<support::endianness target_endianness, bool is64Bits>
SectionRef ELFObjectFile<target_endianness, is64Bits>
                        ::getSectionNext(DataRefImpl Sec) const {
  const uint8_t *sec = *reinterpret_cast<const uint8_t **>(&Sec);
  sec += Header->e_shentsize;
  return SectionRef(DataRefImpl(sec), this);
}

template<support::endianness target_endianness, bool is64Bits>
StringRef ELFObjectFile<target_endianness, is64Bits>
                       ::getSectionName(DataRefImpl Sec) const {
  const Elf_Shdr *sec =
    *reinterpret_cast<const Elf_Shdr **>(&Sec);
  return StringRef(getString(dot_shstrtab_sec, sec->sh_name));
}

template<support::endianness target_endianness, bool is64Bits>
uint64_t ELFObjectFile<target_endianness, is64Bits>
                      ::getSectionAddress(DataRefImpl Sec) const {
  const Elf_Shdr *sec =
    *reinterpret_cast<const Elf_Shdr **>(&Sec);
  return sec->sh_addr;
}

template<support::endianness target_endianness, bool is64Bits>
uint64_t ELFObjectFile<target_endianness, is64Bits>
                      ::getSectionSize(DataRefImpl Sec) const {
  const Elf_Shdr *sec =
    *reinterpret_cast<const Elf_Shdr **>(&Sec);
  return sec->sh_size;
}

template<support::endianness target_endianness, bool is64Bits>
StringRef ELFObjectFile<target_endianness, is64Bits>
                       ::getSectionContents(DataRefImpl Sec) const {
  const Elf_Shdr *sec =
    *reinterpret_cast<const Elf_Shdr **>(&Sec);
  const char *start = (char*)base + sec->sh_offset;
  return StringRef(start, sec->sh_size);
}

template<support::endianness target_endianness, bool is64Bits>
bool ELFObjectFile<target_endianness, is64Bits>
                  ::isSectionText(DataRefImpl Sec) const {
  const Elf_Shdr *sec =
    *reinterpret_cast<const Elf_Shdr **>(&Sec);
  if (sec->sh_flags & ELF::SHF_EXECINSTR)
    return true;
  return false;
}

template<support::endianness target_endianness, bool is64Bits>
ELFObjectFile<target_endianness, is64Bits>::ELFObjectFile(MemoryBuffer *Object)
  : ObjectFile(Object)
  , SectionHeaderTable(0)
  , dot_shstrtab_sec(0)
  , dot_strtab_sec(0) {
  Header = reinterpret_cast<const Elf_Ehdr *>(base);

  if (Header->e_shoff == 0)
    return;

  SectionHeaderTable =
    reinterpret_cast<const Elf_Shdr *>(base + Header->e_shoff);
  uint32_t SectionTableSize = Header->e_shnum * Header->e_shentsize;
  if (!(  (const uint8_t *)SectionHeaderTable + SectionTableSize
         <= base + MapFile->getBufferSize()))
    // FIXME: Proper error handling.
    report_fatal_error("Section table goes past end of file!");


  // To find the symbol tables we walk the section table to find SHT_STMTAB.
  for (const char *i = reinterpret_cast<const char *>(SectionHeaderTable),
                  *e = i + Header->e_shnum * Header->e_shentsize;
                   i != e; i += Header->e_shentsize) {
    const Elf_Shdr *sh = reinterpret_cast<const Elf_Shdr*>(i);
    if (sh->sh_type == ELF::SHT_SYMTAB) {
      SymbolTableSections.push_back(sh);
    }
  }

  // Get string table sections.
  dot_shstrtab_sec = getSection(Header->e_shstrndx);
  if (dot_shstrtab_sec) {
    // Verify that the last byte in the string table in a null.
    if (((const char*)base + dot_shstrtab_sec->sh_offset)
        [dot_shstrtab_sec->sh_size - 1] != 0)
      // FIXME: Proper error handling.
      report_fatal_error("String table must end with a null terminator!");
  }

  // Merge this into the above loop.
  for (const char *i = reinterpret_cast<const char *>(SectionHeaderTable),
                  *e = i + Header->e_shnum * Header->e_shentsize;
                   i != e; i += Header->e_shentsize) {
    const Elf_Shdr *sh = reinterpret_cast<const Elf_Shdr*>(i);
    if (sh->sh_type == ELF::SHT_STRTAB) {
      StringRef SectionName(getString(dot_shstrtab_sec, sh->sh_name));
      if (SectionName == ".strtab") {
        if (dot_strtab_sec != 0)
          // FIXME: Proper error handling.
          report_fatal_error("Already found section named .strtab!");
        dot_strtab_sec = sh;
        const char *dot_strtab = (const char*)base + sh->sh_offset;
          if (dot_strtab[sh->sh_size - 1] != 0)
            // FIXME: Proper error handling.
            report_fatal_error("String table must end with a null terminator!");
      }
    }
  }
}

template<support::endianness target_endianness, bool is64Bits>
ObjectFile::symbol_iterator ELFObjectFile<target_endianness, is64Bits>
                                         ::begin_symbols() const {
  ELFDataRefImpl SymbolData;
  memset(&SymbolData, 0, sizeof(SymbolData));
  if (SymbolTableSections.size() == 0) {
    SymbolData.SymbolIndex = std::numeric_limits<uint32_t>::max();
    SymbolData.SymbolTableSectionIndex = std::numeric_limits<uint32_t>::max();
  } else {
    SymbolData.SymbolIndex = 1; // The 0th symbol in ELF is fake.
    SymbolData.SymbolTableSectionIndex = 0;
  }
  return symbol_iterator(
    SymbolRef(DataRefImpl(*reinterpret_cast<DataRefImpl*>(&SymbolData)), this));
}

template<support::endianness target_endianness, bool is64Bits>
ObjectFile::symbol_iterator ELFObjectFile<target_endianness, is64Bits>
                                         ::end_symbols() const {
  ELFDataRefImpl SymbolData;
  memset(&SymbolData, 0, sizeof(SymbolData));
  SymbolData.SymbolIndex = std::numeric_limits<uint32_t>::max();
  SymbolData.SymbolTableSectionIndex = std::numeric_limits<uint32_t>::max();
  return symbol_iterator(
    SymbolRef(DataRefImpl(*reinterpret_cast<DataRefImpl*>(&SymbolData)), this));
}

template<support::endianness target_endianness, bool is64Bits>
ObjectFile::section_iterator ELFObjectFile<target_endianness, is64Bits>
                                          ::begin_sections() const {
  return section_iterator(
    SectionRef(DataRefImpl(base + Header->e_shoff), this));
}

template<support::endianness target_endianness, bool is64Bits>
ObjectFile::section_iterator ELFObjectFile<target_endianness, is64Bits>
                                          ::end_sections() const {
  return section_iterator(
    SectionRef(DataRefImpl(base
                           + Header->e_shoff
                           + (Header->e_shentsize * Header->e_shnum)), this));
}

template<support::endianness target_endianness, bool is64Bits>
uint8_t ELFObjectFile<target_endianness, is64Bits>::getBytesInAddress() const {
  return 4;
}

template<support::endianness target_endianness, bool is64Bits>
StringRef ELFObjectFile<target_endianness, is64Bits>
                       ::getFileFormatName() const {
  switch(Header->e_ident[ELF::EI_CLASS]) {
  case ELF::ELFCLASS32:
    switch(Header->e_machine) {
    case ELF::EM_386:
      return "ELF32-i386";
    case ELF::EM_X86_64:
      return "ELF32-x86-64";
    default:
      return "ELF32-unknown";
    }
  case ELF::ELFCLASS64:
    switch(Header->e_machine) {
    case ELF::EM_386:
      return "ELF64-i386";
    case ELF::EM_X86_64:
      return "ELF64-x86-64";
    default:
      return "ELF64-unknown";
    }
  default:
    // FIXME: Proper error handling.
    report_fatal_error("Invalid ELFCLASS!");
  }
}

template<support::endianness target_endianness, bool is64Bits>
unsigned ELFObjectFile<target_endianness, is64Bits>::getArch() const {
  switch(Header->e_machine) {
  case ELF::EM_386:
    return Triple::x86;
  case ELF::EM_X86_64:
    return Triple::x86_64;
  default:
    return Triple::UnknownArch;
  }
}

template<support::endianness target_endianness, bool is64Bits>
const typename ELFObjectFile<target_endianness, is64Bits>::Elf_Sym *
ELFObjectFile<target_endianness, is64Bits>::getSymbol(DataRefImpl Symb) const {
  const ELFDataRefImpl SymbolData = *reinterpret_cast<ELFDataRefImpl *>(&Symb);
  const Elf_Shdr *sec =
    SymbolTableSections[SymbolData.SymbolTableSectionIndex];
  return reinterpret_cast<const Elf_Sym *>(
           base
           + sec->sh_offset
           + (SymbolData.SymbolIndex * sec->sh_entsize));
}

template<support::endianness target_endianness, bool is64Bits>
const typename ELFObjectFile<target_endianness, is64Bits>::Elf_Shdr *
ELFObjectFile<target_endianness, is64Bits>::getSection(DataRefImpl Symb) const {
  const ELFDataRefImpl SymbolData = *reinterpret_cast<ELFDataRefImpl *>(&Symb);
  const Elf_Shdr *sec = getSection(SymbolData.SymbolTableSectionIndex);
  if (sec->sh_type != ELF::SHT_SYMTAB)
    // FIXME: Proper error handling.
    report_fatal_error("Invalid symbol table section!");
  return sec;
}

template<support::endianness target_endianness, bool is64Bits>
const typename ELFObjectFile<target_endianness, is64Bits>::Elf_Shdr *
ELFObjectFile<target_endianness, is64Bits>::getSection(uint16_t index) const {
  if (index == 0 || index >= ELF::SHN_LORESERVE)
    return 0;
  if (!SectionHeaderTable || index >= Header->e_shnum)
    // FIXME: Proper error handling.
    report_fatal_error("Invalid section index!");

  return reinterpret_cast<const Elf_Shdr *>(
         reinterpret_cast<const char *>(SectionHeaderTable)
         + (index * Header->e_shentsize));
}

template<support::endianness target_endianness, bool is64Bits>
const char *ELFObjectFile<target_endianness, is64Bits>
                         ::getString(uint16_t section,
                                     ELF::Elf32_Word offset) const {
  return getString(getSection(section), offset);
}

template<support::endianness target_endianness, bool is64Bits>
const char *ELFObjectFile<target_endianness, is64Bits>
                         ::getString(const Elf_Shdr *section,
                                     ELF::Elf32_Word offset) const {
  assert(section && section->sh_type == ELF::SHT_STRTAB && "Invalid section!");
  if (offset >= section->sh_size)
    // FIXME: Proper error handling.
    report_fatal_error("Sybol name offset outside of string table!");
  return (const char *)base + section->sh_offset + offset;
}

// EI_CLASS, EI_DATA.
static std::pair<unsigned char, unsigned char>
getElfArchType(MemoryBuffer *Object) {
  if (Object->getBufferSize() < ELF::EI_NIDENT)
    return std::make_pair((uint8_t)ELF::ELFCLASSNONE,(uint8_t)ELF::ELFDATANONE);
  return std::make_pair( (uint8_t)Object->getBufferStart()[ELF::EI_CLASS]
                       , (uint8_t)Object->getBufferStart()[ELF::EI_DATA]);
}

namespace llvm {

  ObjectFile *ObjectFile::createELFObjectFile(MemoryBuffer *Object) {
    std::pair<unsigned char, unsigned char> Ident = getElfArchType(Object);
    if (Ident.first == ELF::ELFCLASS32 && Ident.second == ELF::ELFDATA2LSB)
      return new ELFObjectFile<support::little, false>(Object);
    else if (Ident.first == ELF::ELFCLASS32 && Ident.second == ELF::ELFDATA2MSB)
      return new ELFObjectFile<support::big, false>(Object);
    else if (Ident.first == ELF::ELFCLASS64 && Ident.second == ELF::ELFDATA2LSB)
      return new ELFObjectFile<support::little, true>(Object);
    else if (Ident.first == ELF::ELFCLASS64 && Ident.second == ELF::ELFDATA2MSB)
      return new ELFObjectFile<support::big, true>(Object);
    // FIXME: Proper error handling.
    report_fatal_error("Not an ELF object file!");
  }

} // end namespace llvm
