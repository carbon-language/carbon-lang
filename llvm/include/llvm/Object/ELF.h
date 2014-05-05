//===- ELF.h - ELF object file implementation -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the ELFFile template class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_ELF_H
#define LLVM_OBJECT_ELF_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/Error.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <limits>
#include <utility>

namespace llvm {
namespace object {

StringRef getELFRelocationTypeName(uint32_t Machine, uint32_t Type);

// Subclasses of ELFFile may need this for template instantiation
inline std::pair<unsigned char, unsigned char>
getElfArchType(MemoryBuffer *Object) {
  if (Object->getBufferSize() < ELF::EI_NIDENT)
    return std::make_pair((uint8_t)ELF::ELFCLASSNONE,(uint8_t)ELF::ELFDATANONE);
  return std::make_pair((uint8_t) Object->getBufferStart()[ELF::EI_CLASS],
                        (uint8_t) Object->getBufferStart()[ELF::EI_DATA]);
}

template <class ELFT>
class ELFFile {
public:
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)
  typedef typename std::conditional<ELFT::Is64Bits,
                                    uint64_t, uint32_t>::type uintX_t;

  /// \brief Iterate over constant sized entities.
  template <class EntT>
  class ELFEntityIterator {
  public:
    typedef ptrdiff_t difference_type;
    typedef EntT value_type;
    typedef std::forward_iterator_tag iterator_category;
    typedef value_type &reference;
    typedef value_type *pointer;

    /// \brief Default construct iterator.
    ELFEntityIterator() : EntitySize(0), Current(nullptr) {}
    ELFEntityIterator(uintX_t EntSize, const char *Start)
        : EntitySize(EntSize), Current(Start) {}

    reference operator *() {
      assert(Current && "Attempted to dereference an invalid iterator!");
      return *reinterpret_cast<pointer>(Current);
    }

    pointer operator ->() {
      assert(Current && "Attempted to dereference an invalid iterator!");
      return reinterpret_cast<pointer>(Current);
    }

    bool operator ==(const ELFEntityIterator &Other) {
      return Current == Other.Current;
    }

    bool operator !=(const ELFEntityIterator &Other) {
      return !(*this == Other);
    }

    ELFEntityIterator &operator ++() {
      assert(Current && "Attempted to increment an invalid iterator!");
      Current += EntitySize;
      return *this;
    }

    ELFEntityIterator operator ++(int) {
      ELFEntityIterator Tmp = *this;
      ++*this;
      return Tmp;
    }

    ELFEntityIterator &operator =(const ELFEntityIterator &Other) {
      EntitySize = Other.EntitySize;
      Current = Other.Current;
      return *this;
    }

    difference_type operator -(const ELFEntityIterator &Other) const {
      assert(EntitySize == Other.EntitySize &&
             "Subtracting iterators of different EntitySize!");
      return (Current - Other.Current) / EntitySize;
    }

    const char *get() const { return Current; }

    uintX_t getEntSize() const { return EntitySize; }

  private:
    uintX_t EntitySize;
    const char *Current;
  };

  typedef Elf_Ehdr_Impl<ELFT> Elf_Ehdr;
  typedef Elf_Shdr_Impl<ELFT> Elf_Shdr;
  typedef Elf_Sym_Impl<ELFT> Elf_Sym;
  typedef Elf_Dyn_Impl<ELFT> Elf_Dyn;
  typedef Elf_Phdr_Impl<ELFT> Elf_Phdr;
  typedef Elf_Rel_Impl<ELFT, false> Elf_Rel;
  typedef Elf_Rel_Impl<ELFT, true> Elf_Rela;
  typedef Elf_Verdef_Impl<ELFT> Elf_Verdef;
  typedef Elf_Verdaux_Impl<ELFT> Elf_Verdaux;
  typedef Elf_Verneed_Impl<ELFT> Elf_Verneed;
  typedef Elf_Vernaux_Impl<ELFT> Elf_Vernaux;
  typedef Elf_Versym_Impl<ELFT> Elf_Versym;
  typedef ELFEntityIterator<const Elf_Dyn> Elf_Dyn_Iter;
  typedef ELFEntityIterator<const Elf_Rela> Elf_Rela_Iter;
  typedef ELFEntityIterator<const Elf_Rel> Elf_Rel_Iter;
  typedef ELFEntityIterator<const Elf_Shdr> Elf_Shdr_Iter;
  typedef iterator_range<Elf_Shdr_Iter> Elf_Shdr_Range;

  /// \brief Archive files are 2 byte aligned, so we need this for
  ///     PointerIntPair to work.
  template <typename T>
  class ArchivePointerTypeTraits {
  public:
    static inline const void *getAsVoidPointer(T *P) { return P; }
    static inline T *getFromVoidPointer(const void *P) {
      return static_cast<T *>(P);
    }
    enum { NumLowBitsAvailable = 1 };
  };

  class Elf_Sym_Iter {
  public:
    typedef ptrdiff_t difference_type;
    typedef const Elf_Sym value_type;
    typedef std::random_access_iterator_tag iterator_category;
    typedef value_type &reference;
    typedef value_type *pointer;

    /// \brief Default construct iterator.
    Elf_Sym_Iter() : EntitySize(0), Current(0, false) {}
    Elf_Sym_Iter(uintX_t EntSize, const char *Start, bool IsDynamic)
        : EntitySize(EntSize), Current(Start, IsDynamic) {}

    reference operator*() {
      assert(Current.getPointer() &&
             "Attempted to dereference an invalid iterator!");
      return *reinterpret_cast<pointer>(Current.getPointer());
    }

    pointer operator->() {
      assert(Current.getPointer() &&
             "Attempted to dereference an invalid iterator!");
      return reinterpret_cast<pointer>(Current.getPointer());
    }

    bool operator==(const Elf_Sym_Iter &Other) {
      return Current == Other.Current;
    }

    bool operator!=(const Elf_Sym_Iter &Other) { return !(*this == Other); }

    Elf_Sym_Iter &operator++() {
      assert(Current.getPointer() &&
             "Attempted to increment an invalid iterator!");
      Current.setPointer(Current.getPointer() + EntitySize);
      return *this;
    }

    Elf_Sym_Iter operator++(int) {
      Elf_Sym_Iter Tmp = *this;
      ++*this;
      return Tmp;
    }

    Elf_Sym_Iter operator+(difference_type Dist) {
      assert(Current.getPointer() &&
             "Attempted to increment an invalid iterator!");
      Current.setPointer(Current.getPointer() + EntitySize * Dist);
      return *this;
    }

    Elf_Sym_Iter &operator=(const Elf_Sym_Iter &Other) {
      EntitySize = Other.EntitySize;
      Current = Other.Current;
      return *this;
    }

    difference_type operator-(const Elf_Sym_Iter &Other) const {
      assert(EntitySize == Other.EntitySize &&
             "Subtracting iterators of different EntitySize!");
      return (Current.getPointer() - Other.Current.getPointer()) / EntitySize;
    }

    const char *get() const { return Current.getPointer(); }

    bool isDynamic() const { return Current.getInt(); }

    uintX_t getEntSize() const { return EntitySize; }

  private:
    uintX_t EntitySize;
    PointerIntPair<const char *, 1, bool,
                   ArchivePointerTypeTraits<const char> > Current;
  };

private:
  typedef SmallVector<const Elf_Shdr *, 2> Sections_t;
  typedef DenseMap<unsigned, unsigned> IndexMap_t;

  MemoryBuffer *Buf;

  const uint8_t *base() const {
    return reinterpret_cast<const uint8_t *>(Buf->getBufferStart());
  }

  const Elf_Ehdr *Header;
  const Elf_Shdr *SectionHeaderTable;
  const Elf_Shdr *dot_shstrtab_sec; // Section header string table.
  const Elf_Shdr *dot_strtab_sec;   // Symbol header string table.
  const Elf_Shdr *dot_symtab_sec;   // Symbol table section.

  const Elf_Shdr *SymbolTableSectionHeaderIndex;
  DenseMap<const Elf_Sym *, ELF::Elf64_Word> ExtendedSymbolTable;

  const Elf_Shdr *dot_gnu_version_sec;   // .gnu.version
  const Elf_Shdr *dot_gnu_version_r_sec; // .gnu.version_r
  const Elf_Shdr *dot_gnu_version_d_sec; // .gnu.version_d

  /// \brief Represents a region described by entries in the .dynamic table.
  struct DynRegionInfo {
    DynRegionInfo() : Addr(nullptr), Size(0), EntSize(0) {}
    /// \brief Address in current address space.
    const void *Addr;
    /// \brief Size in bytes of the region.
    uintX_t Size;
    /// \brief Size of each entity in the region.
    uintX_t EntSize;
  };

  DynRegionInfo DynamicRegion;
  DynRegionInfo DynHashRegion;
  DynRegionInfo DynStrRegion;
  DynRegionInfo DynSymRegion;

  // Pointer to SONAME entry in dynamic string table
  // This is set the first time getLoadName is called.
  mutable const char *dt_soname;

  // Records for each version index the corresponding Verdef or Vernaux entry.
  // This is filled the first time LoadVersionMap() is called.
  class VersionMapEntry : public PointerIntPair<const void*, 1> {
    public:
    // If the integer is 0, this is an Elf_Verdef*.
    // If the integer is 1, this is an Elf_Vernaux*.
    VersionMapEntry() : PointerIntPair<const void*, 1>(nullptr, 0) { }
    VersionMapEntry(const Elf_Verdef *verdef)
        : PointerIntPair<const void*, 1>(verdef, 0) { }
    VersionMapEntry(const Elf_Vernaux *vernaux)
        : PointerIntPair<const void*, 1>(vernaux, 1) { }
    bool isNull() const { return getPointer() == nullptr; }
    bool isVerdef() const { return !isNull() && getInt() == 0; }
    bool isVernaux() const { return !isNull() && getInt() == 1; }
    const Elf_Verdef *getVerdef() const {
      return isVerdef() ? (const Elf_Verdef*)getPointer() : nullptr;
    }
    const Elf_Vernaux *getVernaux() const {
      return isVernaux() ? (const Elf_Vernaux*)getPointer() : nullptr;
    }
  };
  mutable SmallVector<VersionMapEntry, 16> VersionMap;
  void LoadVersionDefs(const Elf_Shdr *sec) const;
  void LoadVersionNeeds(const Elf_Shdr *ec) const;
  void LoadVersionMap() const;

public:
  template<typename T>
  const T        *getEntry(uint32_t Section, uint32_t Entry) const;
  template <typename T>
  const T *getEntry(const Elf_Shdr *Section, uint32_t Entry) const;
  const char     *getString(uint32_t section, uint32_t offset) const;
  const char     *getString(const Elf_Shdr *section, uint32_t offset) const;
  const char *getDynamicString(uintX_t Offset) const;
  ErrorOr<StringRef> getSymbolVersion(const Elf_Shdr *section,
                                      const Elf_Sym *Symb,
                                      bool &IsDefault) const;
  void VerifyStrTab(const Elf_Shdr *sh) const;

  StringRef getRelocationTypeName(uint32_t Type) const;
  void getRelocationTypeName(uint32_t Type,
                             SmallVectorImpl<char> &Result) const;

  /// \brief Get the symbol table section and symbol for a given relocation.
  template <class RelT>
  std::pair<const Elf_Shdr *, const Elf_Sym *>
  getRelocationSymbol(const Elf_Shdr *RelSec, const RelT *Rel) const;

  ELFFile(MemoryBuffer *Object, error_code &ec);

  bool isMipsELF64() const {
    return Header->e_machine == ELF::EM_MIPS &&
      Header->getFileClass() == ELF::ELFCLASS64;
  }

  bool isMips64EL() const {
    return Header->e_machine == ELF::EM_MIPS &&
      Header->getFileClass() == ELF::ELFCLASS64 &&
      Header->getDataEncoding() == ELF::ELFDATA2LSB;
  }

  Elf_Shdr_Iter begin_sections() const;
  Elf_Shdr_Iter end_sections() const;
  Elf_Shdr_Range sections() const {
    return make_range(begin_sections(), end_sections());
  }

  Elf_Sym_Iter begin_symbols() const;
  Elf_Sym_Iter end_symbols() const;

  Elf_Dyn_Iter begin_dynamic_table() const;
  /// \param NULLEnd use one past the first DT_NULL entry as the end instead of
  /// the section size.
  Elf_Dyn_Iter end_dynamic_table(bool NULLEnd = false) const;

  Elf_Sym_Iter begin_dynamic_symbols() const {
    if (DynSymRegion.Addr)
      return Elf_Sym_Iter(DynSymRegion.EntSize, (const char *)DynSymRegion.Addr,
                          true);
    return Elf_Sym_Iter(0, nullptr, true);
  }

  Elf_Sym_Iter end_dynamic_symbols() const {
    if (DynSymRegion.Addr)
      return Elf_Sym_Iter(DynSymRegion.EntSize,
                          (const char *)DynSymRegion.Addr + DynSymRegion.Size,
                          true);
    return Elf_Sym_Iter(0, nullptr, true);
  }

  Elf_Rela_Iter begin_rela(const Elf_Shdr *sec) const {
    return Elf_Rela_Iter(sec->sh_entsize,
                         (const char *)(base() + sec->sh_offset));
  }

  Elf_Rela_Iter end_rela(const Elf_Shdr *sec) const {
    return Elf_Rela_Iter(
        sec->sh_entsize,
        (const char *)(base() + sec->sh_offset + sec->sh_size));
  }

  Elf_Rel_Iter begin_rel(const Elf_Shdr *sec) const {
    return Elf_Rel_Iter(sec->sh_entsize,
                        (const char *)(base() + sec->sh_offset));
  }

  Elf_Rel_Iter end_rel(const Elf_Shdr *sec) const {
    return Elf_Rel_Iter(sec->sh_entsize,
                        (const char *)(base() + sec->sh_offset + sec->sh_size));
  }

  /// \brief Iterate over program header table.
  typedef ELFEntityIterator<const Elf_Phdr> Elf_Phdr_Iter;

  Elf_Phdr_Iter begin_program_headers() const {
    return Elf_Phdr_Iter(Header->e_phentsize,
                         (const char*)base() + Header->e_phoff);
  }

  Elf_Phdr_Iter end_program_headers() const {
    return Elf_Phdr_Iter(Header->e_phentsize,
                         (const char*)base() +
                           Header->e_phoff +
                           (Header->e_phnum * Header->e_phentsize));
  }

  uint64_t getNumSections() const;
  uintX_t getStringTableIndex() const;
  ELF::Elf64_Word getSymbolTableIndex(const Elf_Sym *symb) const;
  const Elf_Ehdr *getHeader() const { return Header; }
  const Elf_Shdr *getSection(const Elf_Sym *symb) const;
  const Elf_Shdr *getSection(uint32_t Index) const;
  const Elf_Sym *getSymbol(uint32_t index) const;

  ErrorOr<StringRef> getSymbolName(Elf_Sym_Iter Sym) const;

  /// \brief Get the name of \p Symb.
  /// \param SymTab The symbol table section \p Symb is contained in.
  /// \param Symb The symbol to get the name of.
  ///
  /// \p SymTab is used to lookup the string table to use to get the symbol's
  /// name.
  ErrorOr<StringRef> getSymbolName(const Elf_Shdr *SymTab,
                                   const Elf_Sym *Symb) const;
  ErrorOr<StringRef> getSectionName(const Elf_Shdr *Section) const;
  uint64_t getSymbolIndex(const Elf_Sym *sym) const;
  ErrorOr<ArrayRef<uint8_t> > getSectionContents(const Elf_Shdr *Sec) const;
  StringRef getLoadName() const;
};

// Use an alignment of 2 for the typedefs since that is the worst case for
// ELF files in archives.
typedef ELFFile<ELFType<support::little, 2, false> > ELF32LEFile;
typedef ELFFile<ELFType<support::little, 2, true> > ELF64LEFile;
typedef ELFFile<ELFType<support::big, 2, false> > ELF32BEFile;
typedef ELFFile<ELFType<support::big, 2, true> > ELF64BEFile;

// Iterate through the version definitions, and place each Elf_Verdef
// in the VersionMap according to its index.
template <class ELFT>
void ELFFile<ELFT>::LoadVersionDefs(const Elf_Shdr *sec) const {
  unsigned vd_size = sec->sh_size;  // Size of section in bytes
  unsigned vd_count = sec->sh_info; // Number of Verdef entries
  const char *sec_start = (const char*)base() + sec->sh_offset;
  const char *sec_end = sec_start + vd_size;
  // The first Verdef entry is at the start of the section.
  const char *p = sec_start;
  for (unsigned i = 0; i < vd_count; i++) {
    if (p + sizeof(Elf_Verdef) > sec_end)
      report_fatal_error("Section ended unexpectedly while scanning "
                         "version definitions.");
    const Elf_Verdef *vd = reinterpret_cast<const Elf_Verdef *>(p);
    if (vd->vd_version != ELF::VER_DEF_CURRENT)
      report_fatal_error("Unexpected verdef version");
    size_t index = vd->vd_ndx & ELF::VERSYM_VERSION;
    if (index >= VersionMap.size())
      VersionMap.resize(index + 1);
    VersionMap[index] = VersionMapEntry(vd);
    p += vd->vd_next;
  }
}

// Iterate through the versions needed section, and place each Elf_Vernaux
// in the VersionMap according to its index.
template <class ELFT>
void ELFFile<ELFT>::LoadVersionNeeds(const Elf_Shdr *sec) const {
  unsigned vn_size = sec->sh_size;  // Size of section in bytes
  unsigned vn_count = sec->sh_info; // Number of Verneed entries
  const char *sec_start = (const char *)base() + sec->sh_offset;
  const char *sec_end = sec_start + vn_size;
  // The first Verneed entry is at the start of the section.
  const char *p = sec_start;
  for (unsigned i = 0; i < vn_count; i++) {
    if (p + sizeof(Elf_Verneed) > sec_end)
      report_fatal_error("Section ended unexpectedly while scanning "
                         "version needed records.");
    const Elf_Verneed *vn = reinterpret_cast<const Elf_Verneed *>(p);
    if (vn->vn_version != ELF::VER_NEED_CURRENT)
      report_fatal_error("Unexpected verneed version");
    // Iterate through the Vernaux entries
    const char *paux = p + vn->vn_aux;
    for (unsigned j = 0; j < vn->vn_cnt; j++) {
      if (paux + sizeof(Elf_Vernaux) > sec_end)
        report_fatal_error("Section ended unexpected while scanning auxiliary "
                           "version needed records.");
      const Elf_Vernaux *vna = reinterpret_cast<const Elf_Vernaux *>(paux);
      size_t index = vna->vna_other & ELF::VERSYM_VERSION;
      if (index >= VersionMap.size())
        VersionMap.resize(index + 1);
      VersionMap[index] = VersionMapEntry(vna);
      paux += vna->vna_next;
    }
    p += vn->vn_next;
  }
}

template <class ELFT>
void ELFFile<ELFT>::LoadVersionMap() const {
  // If there is no dynamic symtab or version table, there is nothing to do.
  if (!DynSymRegion.Addr || !dot_gnu_version_sec)
    return;

  // Has the VersionMap already been loaded?
  if (VersionMap.size() > 0)
    return;

  // The first two version indexes are reserved.
  // Index 0 is LOCAL, index 1 is GLOBAL.
  VersionMap.push_back(VersionMapEntry());
  VersionMap.push_back(VersionMapEntry());

  if (dot_gnu_version_d_sec)
    LoadVersionDefs(dot_gnu_version_d_sec);

  if (dot_gnu_version_r_sec)
    LoadVersionNeeds(dot_gnu_version_r_sec);
}

template <class ELFT>
ELF::Elf64_Word ELFFile<ELFT>::getSymbolTableIndex(const Elf_Sym *symb) const {
  if (symb->st_shndx == ELF::SHN_XINDEX)
    return ExtendedSymbolTable.lookup(symb);
  return symb->st_shndx;
}

template <class ELFT>
const typename ELFFile<ELFT>::Elf_Shdr *
ELFFile<ELFT>::getSection(const Elf_Sym *symb) const {
  if (symb->st_shndx == ELF::SHN_XINDEX)
    return getSection(ExtendedSymbolTable.lookup(symb));
  if (symb->st_shndx >= ELF::SHN_LORESERVE)
    return nullptr;
  return getSection(symb->st_shndx);
}

template <class ELFT>
const typename ELFFile<ELFT>::Elf_Sym *
ELFFile<ELFT>::getSymbol(uint32_t Index) const {
  return &*(begin_symbols() + Index);
}

template <class ELFT>
ErrorOr<ArrayRef<uint8_t> >
ELFFile<ELFT>::getSectionContents(const Elf_Shdr *Sec) const {
  if (Sec->sh_offset + Sec->sh_size > Buf->getBufferSize())
    return object_error::parse_failed;
  const uint8_t *Start = base() + Sec->sh_offset;
  return ArrayRef<uint8_t>(Start, Sec->sh_size);
}

template <class ELFT>
StringRef ELFFile<ELFT>::getRelocationTypeName(uint32_t Type) const {
  return getELFRelocationTypeName(Header->e_machine, Type);
}

template <class ELFT>
void ELFFile<ELFT>::getRelocationTypeName(uint32_t Type,
                                          SmallVectorImpl<char> &Result) const {
  if (!isMipsELF64()) {
    StringRef Name = getRelocationTypeName(Type);
    Result.append(Name.begin(), Name.end());
  } else {
    // The Mips N64 ABI allows up to three operations to be specified per
    // relocation record. Unfortunately there's no easy way to test for the
    // presence of N64 ELFs as they have no special flag that identifies them
    // as being N64. We can safely assume at the moment that all Mips
    // ELFCLASS64 ELFs are N64. New Mips64 ABIs should provide enough
    // information to disambiguate between old vs new ABIs.
    uint8_t Type1 = (Type >> 0) & 0xFF;
    uint8_t Type2 = (Type >> 8) & 0xFF;
    uint8_t Type3 = (Type >> 16) & 0xFF;

    // Concat all three relocation type names.
    StringRef Name = getRelocationTypeName(Type1);
    Result.append(Name.begin(), Name.end());

    Name = getRelocationTypeName(Type2);
    Result.append(1, '/');
    Result.append(Name.begin(), Name.end());

    Name = getRelocationTypeName(Type3);
    Result.append(1, '/');
    Result.append(Name.begin(), Name.end());
  }
}

template <class ELFT>
template <class RelT>
std::pair<const typename ELFFile<ELFT>::Elf_Shdr *,
          const typename ELFFile<ELFT>::Elf_Sym *>
ELFFile<ELFT>::getRelocationSymbol(const Elf_Shdr *Sec, const RelT *Rel) const {
  if (!Sec->sh_link)
    return std::make_pair(nullptr, nullptr);
  const Elf_Shdr *SymTable = getSection(Sec->sh_link);
  return std::make_pair(
      SymTable, getEntry<Elf_Sym>(SymTable, Rel->getSymbol(isMips64EL())));
}

// Verify that the last byte in the string table in a null.
template <class ELFT>
void ELFFile<ELFT>::VerifyStrTab(const Elf_Shdr *sh) const {
  const char *strtab = (const char *)base() + sh->sh_offset;
  if (strtab[sh->sh_size - 1] != 0)
    // FIXME: Proper error handling.
    report_fatal_error("String table must end with a null terminator!");
}

template <class ELFT>
uint64_t ELFFile<ELFT>::getNumSections() const {
  assert(Header && "Header not initialized!");
  if (Header->e_shnum == ELF::SHN_UNDEF) {
    assert(SectionHeaderTable && "SectionHeaderTable not initialized!");
    return SectionHeaderTable->sh_size;
  }
  return Header->e_shnum;
}

template <class ELFT>
typename ELFFile<ELFT>::uintX_t ELFFile<ELFT>::getStringTableIndex() const {
  if (Header->e_shnum == ELF::SHN_UNDEF) {
    if (Header->e_shstrndx == ELF::SHN_HIRESERVE)
      return SectionHeaderTable->sh_link;
    if (Header->e_shstrndx >= getNumSections())
      return 0;
  }
  return Header->e_shstrndx;
}

template <class ELFT>
ELFFile<ELFT>::ELFFile(MemoryBuffer *Object, error_code &ec)
    : Buf(Object),
      SectionHeaderTable(nullptr),
      dot_shstrtab_sec(nullptr),
      dot_strtab_sec(nullptr),
      dot_symtab_sec(nullptr),
      SymbolTableSectionHeaderIndex(nullptr),
      dot_gnu_version_sec(nullptr),
      dot_gnu_version_r_sec(nullptr),
      dot_gnu_version_d_sec(nullptr),
      dt_soname(nullptr) {
  const uint64_t FileSize = Buf->getBufferSize();

  if (sizeof(Elf_Ehdr) > FileSize)
    // FIXME: Proper error handling.
    report_fatal_error("File too short!");

  Header = reinterpret_cast<const Elf_Ehdr *>(base());

  if (Header->e_shoff == 0)
    return;

  const uint64_t SectionTableOffset = Header->e_shoff;

  if (SectionTableOffset + sizeof(Elf_Shdr) > FileSize)
    // FIXME: Proper error handling.
    report_fatal_error("Section header table goes past end of file!");

  // The getNumSections() call below depends on SectionHeaderTable being set.
  SectionHeaderTable =
    reinterpret_cast<const Elf_Shdr *>(base() + SectionTableOffset);
  const uint64_t SectionTableSize = getNumSections() * Header->e_shentsize;

  if (SectionTableOffset + SectionTableSize > FileSize)
    // FIXME: Proper error handling.
    report_fatal_error("Section table goes past end of file!");

  // Scan sections for special sections.

  for (const Elf_Shdr &Sec : sections()) {
    switch (Sec.sh_type) {
    case ELF::SHT_SYMTAB_SHNDX:
      if (SymbolTableSectionHeaderIndex)
        // FIXME: Proper error handling.
        report_fatal_error("More than one .symtab_shndx!");
      SymbolTableSectionHeaderIndex = &Sec;
      break;
    case ELF::SHT_SYMTAB:
      if (dot_symtab_sec)
        // FIXME: Proper error handling.
        report_fatal_error("More than one .symtab!");
      dot_symtab_sec = &Sec;
      dot_strtab_sec = getSection(Sec.sh_link);
      break;
    case ELF::SHT_DYNSYM: {
      if (DynSymRegion.Addr)
        // FIXME: Proper error handling.
        report_fatal_error("More than one .dynsym!");
      DynSymRegion.Addr = base() + Sec.sh_offset;
      DynSymRegion.Size = Sec.sh_size;
      DynSymRegion.EntSize = Sec.sh_entsize;
      const Elf_Shdr *DynStr = getSection(Sec.sh_link);
      DynStrRegion.Addr = base() + DynStr->sh_offset;
      DynStrRegion.Size = DynStr->sh_size;
      DynStrRegion.EntSize = DynStr->sh_entsize;
      break;
    }
    case ELF::SHT_DYNAMIC:
      if (DynamicRegion.Addr)
        // FIXME: Proper error handling.
        report_fatal_error("More than one .dynamic!");
      DynamicRegion.Addr = base() + Sec.sh_offset;
      DynamicRegion.Size = Sec.sh_size;
      DynamicRegion.EntSize = Sec.sh_entsize;
      break;
    case ELF::SHT_GNU_versym:
      if (dot_gnu_version_sec != nullptr)
        // FIXME: Proper error handling.
        report_fatal_error("More than one .gnu.version section!");
      dot_gnu_version_sec = &Sec;
      break;
    case ELF::SHT_GNU_verdef:
      if (dot_gnu_version_d_sec != nullptr)
        // FIXME: Proper error handling.
        report_fatal_error("More than one .gnu.version_d section!");
      dot_gnu_version_d_sec = &Sec;
      break;
    case ELF::SHT_GNU_verneed:
      if (dot_gnu_version_r_sec != nullptr)
        // FIXME: Proper error handling.
        report_fatal_error("More than one .gnu.version_r section!");
      dot_gnu_version_r_sec = &Sec;
      break;
    }
  }

  // Get string table sections.
  dot_shstrtab_sec = getSection(getStringTableIndex());
  if (dot_shstrtab_sec) {
    // Verify that the last byte in the string table in a null.
    VerifyStrTab(dot_shstrtab_sec);
  }

  // Build symbol name side-mapping if there is one.
  if (SymbolTableSectionHeaderIndex) {
    const Elf_Word *ShndxTable = reinterpret_cast<const Elf_Word*>(base() +
                                      SymbolTableSectionHeaderIndex->sh_offset);
    for (Elf_Sym_Iter SI = begin_symbols(), SE = end_symbols(); SI != SE;
         ++SI) {
      if (*ShndxTable != ELF::SHN_UNDEF)
        ExtendedSymbolTable[&*SI] = *ShndxTable;
      ++ShndxTable;
    }
  }

  // Scan program headers.
  for (Elf_Phdr_Iter PhdrI = begin_program_headers(),
                     PhdrE = end_program_headers();
       PhdrI != PhdrE; ++PhdrI) {
    if (PhdrI->p_type == ELF::PT_DYNAMIC) {
      DynamicRegion.Addr = base() + PhdrI->p_offset;
      DynamicRegion.Size = PhdrI->p_filesz;
      DynamicRegion.EntSize = sizeof(Elf_Dyn);
      break;
    }
  }

  ec = error_code::success();
}

// Get the symbol table index in the symtab section given a symbol
template <class ELFT>
uint64_t ELFFile<ELFT>::getSymbolIndex(const Elf_Sym *Sym) const {
  uintptr_t SymLoc = uintptr_t(Sym);
  uintptr_t SymTabLoc = uintptr_t(base() + dot_symtab_sec->sh_offset);
  assert(SymLoc > SymTabLoc && "Symbol not in symbol table!");
  uint64_t SymOffset = SymLoc - SymTabLoc;
  assert(SymOffset % dot_symtab_sec->sh_entsize == 0 &&
         "Symbol not multiple of symbol size!");
  return SymOffset / dot_symtab_sec->sh_entsize;
}

template <class ELFT>
typename ELFFile<ELFT>::Elf_Shdr_Iter ELFFile<ELFT>::begin_sections() const {
  return Elf_Shdr_Iter(Header->e_shentsize,
                       (const char *)base() + Header->e_shoff);
}

template <class ELFT>
typename ELFFile<ELFT>::Elf_Shdr_Iter ELFFile<ELFT>::end_sections() const {
  return Elf_Shdr_Iter(Header->e_shentsize,
                       (const char *)base() + Header->e_shoff +
                           (getNumSections() * Header->e_shentsize));
}

template <class ELFT>
typename ELFFile<ELFT>::Elf_Sym_Iter ELFFile<ELFT>::begin_symbols() const {
  if (!dot_symtab_sec)
    return Elf_Sym_Iter(0, nullptr, false);
  return Elf_Sym_Iter(dot_symtab_sec->sh_entsize,
                      (const char *)base() + dot_symtab_sec->sh_offset, false);
}

template <class ELFT>
typename ELFFile<ELFT>::Elf_Sym_Iter ELFFile<ELFT>::end_symbols() const {
  if (!dot_symtab_sec)
    return Elf_Sym_Iter(0, nullptr, false);
  return Elf_Sym_Iter(dot_symtab_sec->sh_entsize,
                      (const char *)base() + dot_symtab_sec->sh_offset +
                          dot_symtab_sec->sh_size,
                      false);
}

template <class ELFT>
typename ELFFile<ELFT>::Elf_Dyn_Iter
ELFFile<ELFT>::begin_dynamic_table() const {
  if (DynamicRegion.Addr)
    return Elf_Dyn_Iter(DynamicRegion.EntSize,
                        (const char *)DynamicRegion.Addr);
  return Elf_Dyn_Iter(0, nullptr);
}

template <class ELFT>
typename ELFFile<ELFT>::Elf_Dyn_Iter
ELFFile<ELFT>::end_dynamic_table(bool NULLEnd) const {
  if (!DynamicRegion.Addr)
    return Elf_Dyn_Iter(0, nullptr);
  Elf_Dyn_Iter Ret(DynamicRegion.EntSize,
                    (const char *)DynamicRegion.Addr + DynamicRegion.Size);

  if (NULLEnd) {
    Elf_Dyn_Iter Start = begin_dynamic_table();
    while (Start != Ret && Start->getTag() != ELF::DT_NULL)
      ++Start;

    // Include the DT_NULL.
    if (Start != Ret)
      ++Start;
    Ret = Start;
  }
  return Ret;
}

template <class ELFT>
StringRef ELFFile<ELFT>::getLoadName() const {
  if (!dt_soname) {
    // Find the DT_SONAME entry
    Elf_Dyn_Iter it = begin_dynamic_table();
    Elf_Dyn_Iter ie = end_dynamic_table();
    while (it != ie && it->getTag() != ELF::DT_SONAME)
      ++it;

    if (it != ie) {
      dt_soname = getDynamicString(it->getVal());
    } else {
      dt_soname = "";
    }
  }
  return dt_soname;
}

template <class ELFT>
template <typename T>
const T *ELFFile<ELFT>::getEntry(uint32_t Section, uint32_t Entry) const {
  return getEntry<T>(getSection(Section), Entry);
}

template <class ELFT>
template <typename T>
const T *ELFFile<ELFT>::getEntry(const Elf_Shdr *Section,
                                 uint32_t Entry) const {
  return reinterpret_cast<const T *>(base() + Section->sh_offset +
                                     (Entry * Section->sh_entsize));
}

template <class ELFT>
const typename ELFFile<ELFT>::Elf_Shdr *
ELFFile<ELFT>::getSection(uint32_t index) const {
  if (index == 0)
    return nullptr;
  if (!SectionHeaderTable || index >= getNumSections())
    // FIXME: Proper error handling.
    report_fatal_error("Invalid section index!");

  return reinterpret_cast<const Elf_Shdr *>(
         reinterpret_cast<const char *>(SectionHeaderTable)
         + (index * Header->e_shentsize));
}

template <class ELFT>
const char *ELFFile<ELFT>::getString(uint32_t section,
                                     ELF::Elf32_Word offset) const {
  return getString(getSection(section), offset);
}

template <class ELFT>
const char *ELFFile<ELFT>::getString(const Elf_Shdr *section,
                                     ELF::Elf32_Word offset) const {
  assert(section && section->sh_type == ELF::SHT_STRTAB && "Invalid section!");
  if (offset >= section->sh_size)
    // FIXME: Proper error handling.
    report_fatal_error("Symbol name offset outside of string table!");
  return (const char *)base() + section->sh_offset + offset;
}

template <class ELFT>
const char *ELFFile<ELFT>::getDynamicString(uintX_t Offset) const {
  if (!DynStrRegion.Addr || Offset >= DynStrRegion.Size)
    return nullptr;
  return (const char *)DynStrRegion.Addr + Offset;
}

template <class ELFT>
ErrorOr<StringRef> ELFFile<ELFT>::getSymbolName(Elf_Sym_Iter Sym) const {
  if (!Sym.isDynamic())
    return getSymbolName(dot_symtab_sec, &*Sym);

  if (!DynStrRegion.Addr || Sym->st_name >= DynStrRegion.Size)
    return object_error::parse_failed;
  return StringRef(getDynamicString(Sym->st_name));
}

template <class ELFT>
ErrorOr<StringRef> ELFFile<ELFT>::getSymbolName(const Elf_Shdr *Section,
                                                const Elf_Sym *Symb) const {
  if (Symb->st_name == 0) {
    const Elf_Shdr *ContainingSec = getSection(Symb);
    if (ContainingSec)
      return getSectionName(ContainingSec);
  }

  const Elf_Shdr *StrTab = getSection(Section->sh_link);
  if (Symb->st_name >= StrTab->sh_size)
    return object_error::parse_failed;
  return StringRef(getString(StrTab, Symb->st_name));
}

template <class ELFT>
ErrorOr<StringRef>
ELFFile<ELFT>::getSectionName(const Elf_Shdr *Section) const {
  if (Section->sh_name >= dot_shstrtab_sec->sh_size)
    return object_error::parse_failed;
  return StringRef(getString(dot_shstrtab_sec, Section->sh_name));
}

template <class ELFT>
ErrorOr<StringRef> ELFFile<ELFT>::getSymbolVersion(const Elf_Shdr *section,
                                                   const Elf_Sym *symb,
                                                   bool &IsDefault) const {
  // Handle non-dynamic symbols.
  if (section != DynSymRegion.Addr && section != nullptr) {
    // Non-dynamic symbols can have versions in their names
    // A name of the form 'foo@V1' indicates version 'V1', non-default.
    // A name of the form 'foo@@V2' indicates version 'V2', default version.
    ErrorOr<StringRef> SymName = getSymbolName(section, symb);
    if (!SymName)
      return SymName;
    StringRef Name = *SymName;
    size_t atpos = Name.find('@');
    if (atpos == StringRef::npos) {
      IsDefault = false;
      return StringRef("");
    }
    ++atpos;
    if (atpos < Name.size() && Name[atpos] == '@') {
      IsDefault = true;
      ++atpos;
    } else {
      IsDefault = false;
    }
    return Name.substr(atpos);
  }

  // This is a dynamic symbol. Look in the GNU symbol version table.
  if (!dot_gnu_version_sec) {
    // No version table.
    IsDefault = false;
    return StringRef("");
  }

  // Determine the position in the symbol table of this entry.
  size_t entry_index = ((const char *)symb - (const char *)DynSymRegion.Addr) /
                       DynSymRegion.EntSize;

  // Get the corresponding version index entry
  const Elf_Versym *vs = getEntry<Elf_Versym>(dot_gnu_version_sec, entry_index);
  size_t version_index = vs->vs_index & ELF::VERSYM_VERSION;

  // Special markers for unversioned symbols.
  if (version_index == ELF::VER_NDX_LOCAL ||
      version_index == ELF::VER_NDX_GLOBAL) {
    IsDefault = false;
    return StringRef("");
  }

  // Lookup this symbol in the version table
  LoadVersionMap();
  if (version_index >= VersionMap.size() || VersionMap[version_index].isNull())
    return object_error::parse_failed;
  const VersionMapEntry &entry = VersionMap[version_index];

  // Get the version name string
  size_t name_offset;
  if (entry.isVerdef()) {
    // The first Verdaux entry holds the name.
    name_offset = entry.getVerdef()->getAux()->vda_name;
  } else {
    name_offset = entry.getVernaux()->vna_name;
  }

  // Set IsDefault
  if (entry.isVerdef()) {
    IsDefault = !(vs->vs_index & ELF::VERSYM_HIDDEN);
  } else {
    IsDefault = false;
  }

  if (name_offset >= DynStrRegion.Size)
    return object_error::parse_failed;
  return StringRef(getDynamicString(name_offset));
}

/// This function returns the hash value for a symbol in the .dynsym section
/// Name of the API remains consistent as specified in the libelf
/// REF : http://www.sco.com/developers/gabi/latest/ch5.dynamic.html#hash
static inline unsigned elf_hash(StringRef &symbolName) {
  unsigned h = 0, g;
  for (unsigned i = 0, j = symbolName.size(); i < j; i++) {
    h = (h << 4) + symbolName[i];
    g = h & 0xf0000000L;
    if (g != 0)
      h ^= g >> 24;
    h &= ~g;
  }
  return h;
}
} // end namespace object
} // end namespace llvm

#endif
