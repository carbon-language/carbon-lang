//===- ELF.h - ELF object file implementation -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the ELFObjectFile template class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_ELF_H
#define LLVM_OBJECT_ELF_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <limits>
#include <utility>

namespace llvm {
namespace object {

using support::endianness;

template<endianness target_endianness, std::size_t max_alignment, bool is64Bits>
struct ELFType {
  static const endianness TargetEndianness = target_endianness;
  static const std::size_t MaxAlignment = max_alignment;
  static const bool Is64Bits = is64Bits;
};

template<typename T, int max_align>
struct MaximumAlignment {
  enum {value = AlignOf<T>::Alignment > max_align ? max_align
                                                  : AlignOf<T>::Alignment};
};

// Subclasses of ELFObjectFile may need this for template instantiation
inline std::pair<unsigned char, unsigned char>
getElfArchType(MemoryBuffer *Object) {
  if (Object->getBufferSize() < ELF::EI_NIDENT)
    return std::make_pair((uint8_t)ELF::ELFCLASSNONE,(uint8_t)ELF::ELFDATANONE);
  return std::make_pair( (uint8_t)Object->getBufferStart()[ELF::EI_CLASS]
                       , (uint8_t)Object->getBufferStart()[ELF::EI_DATA]);
}

// Templates to choose Elf_Addr and Elf_Off depending on is64Bits.
template<endianness target_endianness, std::size_t max_alignment>
struct ELFDataTypeTypedefHelperCommon {
  typedef support::detail::packed_endian_specific_integral
    <uint16_t, target_endianness,
     MaximumAlignment<uint16_t, max_alignment>::value> Elf_Half;
  typedef support::detail::packed_endian_specific_integral
    <uint32_t, target_endianness,
     MaximumAlignment<uint32_t, max_alignment>::value> Elf_Word;
  typedef support::detail::packed_endian_specific_integral
    <int32_t, target_endianness,
     MaximumAlignment<int32_t, max_alignment>::value> Elf_Sword;
  typedef support::detail::packed_endian_specific_integral
    <uint64_t, target_endianness,
     MaximumAlignment<uint64_t, max_alignment>::value> Elf_Xword;
  typedef support::detail::packed_endian_specific_integral
    <int64_t, target_endianness,
     MaximumAlignment<int64_t, max_alignment>::value> Elf_Sxword;
};

template<class ELFT>
struct ELFDataTypeTypedefHelper;

/// ELF 32bit types.
template<endianness TargetEndianness, std::size_t MaxAlign>
struct ELFDataTypeTypedefHelper<ELFType<TargetEndianness, MaxAlign, false> >
  : ELFDataTypeTypedefHelperCommon<TargetEndianness, MaxAlign> {
  typedef uint32_t value_type;
  typedef support::detail::packed_endian_specific_integral
    <value_type, TargetEndianness,
     MaximumAlignment<value_type, MaxAlign>::value> Elf_Addr;
  typedef support::detail::packed_endian_specific_integral
    <value_type, TargetEndianness,
     MaximumAlignment<value_type, MaxAlign>::value> Elf_Off;
};

/// ELF 64bit types.
template<endianness TargetEndianness, std::size_t MaxAlign>
struct ELFDataTypeTypedefHelper<ELFType<TargetEndianness, MaxAlign, true> >
  : ELFDataTypeTypedefHelperCommon<TargetEndianness, MaxAlign> {
  typedef uint64_t value_type;
  typedef support::detail::packed_endian_specific_integral
    <value_type, TargetEndianness,
     MaximumAlignment<value_type, MaxAlign>::value> Elf_Addr;
  typedef support::detail::packed_endian_specific_integral
    <value_type, TargetEndianness,
     MaximumAlignment<value_type, MaxAlign>::value> Elf_Off;
};

// I really don't like doing this, but the alternative is copypasta.
#define LLVM_ELF_IMPORT_TYPES(E, M, W)                                         \
typedef typename ELFDataTypeTypedefHelper<ELFType<E,M,W> >::Elf_Addr Elf_Addr; \
typedef typename ELFDataTypeTypedefHelper<ELFType<E,M,W> >::Elf_Off Elf_Off;   \
typedef typename ELFDataTypeTypedefHelper<ELFType<E,M,W> >::Elf_Half Elf_Half; \
typedef typename ELFDataTypeTypedefHelper<ELFType<E,M,W> >::Elf_Word Elf_Word; \
typedef typename                                                               \
  ELFDataTypeTypedefHelper<ELFType<E,M,W> >::Elf_Sword Elf_Sword;              \
typedef typename                                                               \
  ELFDataTypeTypedefHelper<ELFType<E,M,W> >::Elf_Xword Elf_Xword;              \
typedef typename                                                               \
  ELFDataTypeTypedefHelper<ELFType<E,M,W> >::Elf_Sxword Elf_Sxword;

#define LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)                                       \
  LLVM_ELF_IMPORT_TYPES(ELFT::TargetEndianness, ELFT::MaxAlignment,            \
  ELFT::Is64Bits)

// Section header.
template<class ELFT>
struct Elf_Shdr_Base;

template<endianness TargetEndianness, std::size_t MaxAlign>
struct Elf_Shdr_Base<ELFType<TargetEndianness, MaxAlign, false> > {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, MaxAlign, false)
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

template<endianness TargetEndianness, std::size_t MaxAlign>
struct Elf_Shdr_Base<ELFType<TargetEndianness, MaxAlign, true> > {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, MaxAlign, true)
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

template<class ELFT>
struct Elf_Shdr_Impl : Elf_Shdr_Base<ELFT> {
  using Elf_Shdr_Base<ELFT>::sh_entsize;
  using Elf_Shdr_Base<ELFT>::sh_size;

  /// @brief Get the number of entities this section contains if it has any.
  unsigned getEntityCount() const {
    if (sh_entsize == 0)
      return 0;
    return sh_size / sh_entsize;
  }
};

template<class ELFT>
struct Elf_Sym_Base;

template<endianness TargetEndianness, std::size_t MaxAlign>
struct Elf_Sym_Base<ELFType<TargetEndianness, MaxAlign, false> > {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, MaxAlign, false)
  Elf_Word      st_name;  // Symbol name (index into string table)
  Elf_Addr      st_value; // Value or address associated with the symbol
  Elf_Word      st_size;  // Size of the symbol
  unsigned char st_info;  // Symbol's type and binding attributes
  unsigned char st_other; // Must be zero; reserved
  Elf_Half      st_shndx; // Which section (header table index) it's defined in
};

template<endianness TargetEndianness, std::size_t MaxAlign>
struct Elf_Sym_Base<ELFType<TargetEndianness, MaxAlign, true> > {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, MaxAlign, true)
  Elf_Word      st_name;  // Symbol name (index into string table)
  unsigned char st_info;  // Symbol's type and binding attributes
  unsigned char st_other; // Must be zero; reserved
  Elf_Half      st_shndx; // Which section (header table index) it's defined in
  Elf_Addr      st_value; // Value or address associated with the symbol
  Elf_Xword     st_size;  // Size of the symbol
};

template<class ELFT>
struct Elf_Sym_Impl : Elf_Sym_Base<ELFT> {
  using Elf_Sym_Base<ELFT>::st_info;

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

/// Elf_Versym: This is the structure of entries in the SHT_GNU_versym section
/// (.gnu.version). This structure is identical for ELF32 and ELF64.
template<class ELFT>
struct Elf_Versym_Impl {
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)
  Elf_Half vs_index;   // Version index with flags (e.g. VERSYM_HIDDEN)
};

template<class ELFT>
struct Elf_Verdaux_Impl;

/// Elf_Verdef: This is the structure of entries in the SHT_GNU_verdef section
/// (.gnu.version_d). This structure is identical for ELF32 and ELF64.
template<class ELFT>
struct Elf_Verdef_Impl {
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)
  typedef Elf_Verdaux_Impl<ELFT> Elf_Verdaux;
  Elf_Half vd_version; // Version of this structure (e.g. VER_DEF_CURRENT)
  Elf_Half vd_flags;   // Bitwise flags (VER_DEF_*)
  Elf_Half vd_ndx;     // Version index, used in .gnu.version entries
  Elf_Half vd_cnt;     // Number of Verdaux entries
  Elf_Word vd_hash;    // Hash of name
  Elf_Word vd_aux;     // Offset to the first Verdaux entry (in bytes)
  Elf_Word vd_next;    // Offset to the next Verdef entry (in bytes)

  /// Get the first Verdaux entry for this Verdef.
  const Elf_Verdaux *getAux() const {
    return reinterpret_cast<const Elf_Verdaux*>((const char*)this + vd_aux);
  }
};

/// Elf_Verdaux: This is the structure of auxiliary data in the SHT_GNU_verdef
/// section (.gnu.version_d). This structure is identical for ELF32 and ELF64.
template<class ELFT>
struct Elf_Verdaux_Impl {
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)
  Elf_Word vda_name; // Version name (offset in string table)
  Elf_Word vda_next; // Offset to next Verdaux entry (in bytes)
};

/// Elf_Verneed: This is the structure of entries in the SHT_GNU_verneed
/// section (.gnu.version_r). This structure is identical for ELF32 and ELF64.
template<class ELFT>
struct Elf_Verneed_Impl {
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)
  Elf_Half vn_version; // Version of this structure (e.g. VER_NEED_CURRENT)
  Elf_Half vn_cnt;     // Number of associated Vernaux entries
  Elf_Word vn_file;    // Library name (string table offset)
  Elf_Word vn_aux;     // Offset to first Vernaux entry (in bytes)
  Elf_Word vn_next;    // Offset to next Verneed entry (in bytes)
};

/// Elf_Vernaux: This is the structure of auxiliary data in SHT_GNU_verneed
/// section (.gnu.version_r). This structure is identical for ELF32 and ELF64.
template<class ELFT>
struct Elf_Vernaux_Impl {
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)
  Elf_Word vna_hash;  // Hash of dependency name
  Elf_Half vna_flags; // Bitwise Flags (VER_FLAG_*)
  Elf_Half vna_other; // Version index, used in .gnu.version entries
  Elf_Word vna_name;  // Dependency name
  Elf_Word vna_next;  // Offset to next Vernaux entry (in bytes)
};

/// Elf_Dyn_Base: This structure matches the form of entries in the dynamic
///               table section (.dynamic) look like.
template<class ELFT>
struct Elf_Dyn_Base;

template<endianness TargetEndianness, std::size_t MaxAlign>
struct Elf_Dyn_Base<ELFType<TargetEndianness, MaxAlign, false> > {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, MaxAlign, false)
  Elf_Sword d_tag;
  union {
    Elf_Word d_val;
    Elf_Addr d_ptr;
  } d_un;
};

template<endianness TargetEndianness, std::size_t MaxAlign>
struct Elf_Dyn_Base<ELFType<TargetEndianness, MaxAlign, true> > {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, MaxAlign, true)
  Elf_Sxword d_tag;
  union {
    Elf_Xword d_val;
    Elf_Addr d_ptr;
  } d_un;
};

/// Elf_Dyn_Impl: This inherits from Elf_Dyn_Base, adding getters and setters.
template<class ELFT>
struct Elf_Dyn_Impl : Elf_Dyn_Base<ELFT> {
  using Elf_Dyn_Base<ELFT>::d_tag;
  using Elf_Dyn_Base<ELFT>::d_un;
  int64_t getTag() const { return d_tag; }
  uint64_t getVal() const { return d_un.d_val; }
  uint64_t getPtr() const { return d_un.ptr; }
};

// Elf_Rel: Elf Relocation
template<class ELFT, bool isRela>
struct Elf_Rel_Base;

template<endianness TargetEndianness, std::size_t MaxAlign>
struct Elf_Rel_Base<ELFType<TargetEndianness, MaxAlign, false>, false> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, MaxAlign, false)
  Elf_Addr      r_offset; // Location (file byte offset, or program virtual addr)
  Elf_Word      r_info;  // Symbol table index and type of relocation to apply

  uint32_t getRInfo(bool isMips64EL) const {
    assert(!isMips64EL);
    return r_info;
  }
  void setRInfo(uint32_t R) {
    r_info = R;
  }
};

template<endianness TargetEndianness, std::size_t MaxAlign>
struct Elf_Rel_Base<ELFType<TargetEndianness, MaxAlign, true>, false> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, MaxAlign, true)
  Elf_Addr      r_offset; // Location (file byte offset, or program virtual addr)
  Elf_Xword     r_info;   // Symbol table index and type of relocation to apply

  uint64_t getRInfo(bool isMips64EL) const {
    uint64_t t = r_info;
    if (!isMips64EL)
      return t;
    // Mip64 little endian has a "special" encoding of r_info. Instead of one
    // 64 bit little endian number, it is a little ending 32 bit number followed
    // by a 32 bit big endian number.
    return (t << 32) | ((t >> 8) & 0xff000000) | ((t >> 24) & 0x00ff0000) |
      ((t >> 40) & 0x0000ff00) | ((t >> 56) & 0x000000ff);
    return r_info;
  }
  void setRInfo(uint64_t R) {
    // FIXME: Add mips64el support.
    r_info = R;
  }
};

template<endianness TargetEndianness, std::size_t MaxAlign>
struct Elf_Rel_Base<ELFType<TargetEndianness, MaxAlign, false>, true> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, MaxAlign, false)
  Elf_Addr      r_offset; // Location (file byte offset, or program virtual addr)
  Elf_Word      r_info;   // Symbol table index and type of relocation to apply
  Elf_Sword     r_addend; // Compute value for relocatable field by adding this

  uint32_t getRInfo(bool isMips64EL) const {
    assert(!isMips64EL);
    return r_info;
  }
  void setRInfo(uint32_t R) {
    r_info = R;
  }
};

template<endianness TargetEndianness, std::size_t MaxAlign>
struct Elf_Rel_Base<ELFType<TargetEndianness, MaxAlign, true>, true> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, MaxAlign, true)
  Elf_Addr      r_offset; // Location (file byte offset, or program virtual addr)
  Elf_Xword     r_info;   // Symbol table index and type of relocation to apply
  Elf_Sxword    r_addend; // Compute value for relocatable field by adding this.

  uint64_t getRInfo(bool isMips64EL) const {
    // Mip64 little endian has a "special" encoding of r_info. Instead of one
    // 64 bit little endian number, it is a little ending 32 bit number followed
    // by a 32 bit big endian number.
    uint64_t t = r_info;
    if (!isMips64EL)
      return t;
    return (t << 32) | ((t >> 8) & 0xff000000) | ((t >> 24) & 0x00ff0000) |
      ((t >> 40) & 0x0000ff00) | ((t >> 56) & 0x000000ff);
  }
  void setRInfo(uint64_t R) {
    // FIXME: Add mips64el support.
    r_info = R;
  }
};

template<class ELFT, bool isRela>
struct Elf_Rel_Impl;

template<endianness TargetEndianness, std::size_t MaxAlign, bool isRela>
struct Elf_Rel_Impl<ELFType<TargetEndianness, MaxAlign, true>, isRela>
       : Elf_Rel_Base<ELFType<TargetEndianness, MaxAlign, true>, isRela> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, MaxAlign, true)

  // These accessors and mutators correspond to the ELF64_R_SYM, ELF64_R_TYPE,
  // and ELF64_R_INFO macros defined in the ELF specification:
  uint32_t getSymbol(bool isMips64EL) const {
    return (uint32_t) (this->getRInfo(isMips64EL) >> 32);
  }
  uint32_t getType(bool isMips64EL) const {
    return (uint32_t) (this->getRInfo(isMips64EL) & 0xffffffffL);
  }
  void setSymbol(uint32_t s) { setSymbolAndType(s, getType()); }
  void setType(uint32_t t) { setSymbolAndType(getSymbol(), t); }
  void setSymbolAndType(uint32_t s, uint32_t t) {
    this->setRInfo(((uint64_t)s << 32) + (t&0xffffffffL));
  }
};

template<endianness TargetEndianness, std::size_t MaxAlign, bool isRela>
struct Elf_Rel_Impl<ELFType<TargetEndianness, MaxAlign, false>, isRela>
       : Elf_Rel_Base<ELFType<TargetEndianness, MaxAlign, false>, isRela> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, MaxAlign, false)

  // These accessors and mutators correspond to the ELF32_R_SYM, ELF32_R_TYPE,
  // and ELF32_R_INFO macros defined in the ELF specification:
  uint32_t getSymbol(bool isMips64EL) const {
    return this->getRInfo(isMips64EL) >> 8;
  }
  unsigned char getType(bool isMips64EL) const {
    return (unsigned char) (this->getRInfo(isMips64EL) & 0x0ff);
  }
  void setSymbol(uint32_t s) { setSymbolAndType(s, getType()); }
  void setType(unsigned char t) { setSymbolAndType(getSymbol(), t); }
  void setSymbolAndType(uint32_t s, unsigned char t) {
    this->setRInfo((s << 8) + t);
  }
};

template<class ELFT>
struct Elf_Ehdr_Impl {
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)
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

template<class ELFT>
struct Elf_Phdr_Impl;

template<endianness TargetEndianness, std::size_t MaxAlign>
struct Elf_Phdr_Impl<ELFType<TargetEndianness, MaxAlign, false> > {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, MaxAlign, false)
  Elf_Word p_type;   // Type of segment
  Elf_Off  p_offset; // FileOffset where segment is located, in bytes
  Elf_Addr p_vaddr;  // Virtual Address of beginning of segment
  Elf_Addr p_paddr;  // Physical address of beginning of segment (OS-specific)
  Elf_Word p_filesz; // Num. of bytes in file image of segment (may be zero)
  Elf_Word p_memsz;  // Num. of bytes in mem image of segment (may be zero)
  Elf_Word p_flags;  // Segment flags
  Elf_Word p_align;  // Segment alignment constraint
};

template<endianness TargetEndianness, std::size_t MaxAlign>
struct Elf_Phdr_Impl<ELFType<TargetEndianness, MaxAlign, true> > {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, MaxAlign, true)
  Elf_Word p_type;   // Type of segment
  Elf_Word p_flags;  // Segment flags
  Elf_Off  p_offset; // FileOffset where segment is located, in bytes
  Elf_Addr p_vaddr;  // Virtual Address of beginning of segment
  Elf_Addr p_paddr;  // Physical address of beginning of segment (OS-specific)
  Elf_Xword p_filesz; // Num. of bytes in file image of segment (may be zero)
  Elf_Xword p_memsz;  // Num. of bytes in mem image of segment (may be zero)
  Elf_Xword p_align;  // Segment alignment constraint
};

template<class ELFT>
class ELFObjectFile : public ObjectFile {
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)

public:
  /// \brief Iterate over constant sized entities.
  template<class EntT>
  class ELFEntityIterator {
  public:
    typedef ptrdiff_t difference_type;
    typedef EntT value_type;
    typedef std::random_access_iterator_tag iterator_category;
    typedef value_type &reference;
    typedef value_type *pointer;

    /// \brief Default construct iterator.
    ELFEntityIterator() : EntitySize(0), Current(0) {}
    ELFEntityIterator(uint64_t EntSize, const char *Start)
      : EntitySize(EntSize)
      , Current(Start) {}

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
             "Subtracting iterators of different EntitiySize!");
      return (Current - Other.Current) / EntitySize;
    }

    const char *get() const { return Current; }

  private:
    uint64_t EntitySize;
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
  typedef ELFEntityIterator<const Elf_Dyn> Elf_Dyn_iterator;
  typedef ELFEntityIterator<const Elf_Sym> Elf_Sym_iterator;
  typedef ELFEntityIterator<const Elf_Rela> Elf_Rela_Iter;
  typedef ELFEntityIterator<const Elf_Rel> Elf_Rel_Iter;

protected:
  // This flag is used for classof, to distinguish ELFObjectFile from
  // its subclass. If more subclasses will be created, this flag will
  // have to become an enum.
  bool isDyldELFObject;

private:
  typedef SmallVector<const Elf_Shdr *, 2> Sections_t;
  typedef DenseMap<unsigned, unsigned> IndexMap_t;
  typedef DenseMap<const Elf_Shdr*, SmallVector<uint32_t, 1> > RelocMap_t;

  const Elf_Ehdr *Header;
  const Elf_Shdr *SectionHeaderTable;
  const Elf_Shdr *dot_shstrtab_sec; // Section header string table.
  const Elf_Shdr *dot_strtab_sec;   // Symbol header string table.
  const Elf_Shdr *dot_dynstr_sec;   // Dynamic symbol string table.

  // SymbolTableSections[0] always points to the dynamic string table section
  // header, or NULL if there is no dynamic string table.
  Sections_t SymbolTableSections;
  IndexMap_t SymbolTableSectionsIndexMap;
  DenseMap<const Elf_Sym*, ELF::Elf64_Word> ExtendedSymbolTable;

  const Elf_Shdr *dot_dynamic_sec;       // .dynamic
  const Elf_Shdr *dot_gnu_version_sec;   // .gnu.version
  const Elf_Shdr *dot_gnu_version_r_sec; // .gnu.version_r
  const Elf_Shdr *dot_gnu_version_d_sec; // .gnu.version_d

  // Pointer to SONAME entry in dynamic string table
  // This is set the first time getLoadName is called.
  mutable const char *dt_soname;

private:
  uint64_t getROffset(DataRefImpl Rel) const;

  // Records for each version index the corresponding Verdef or Vernaux entry.
  // This is filled the first time LoadVersionMap() is called.
  class VersionMapEntry : public PointerIntPair<const void*, 1> {
    public:
    // If the integer is 0, this is an Elf_Verdef*.
    // If the integer is 1, this is an Elf_Vernaux*.
    VersionMapEntry() : PointerIntPair<const void*, 1>(NULL, 0) { }
    VersionMapEntry(const Elf_Verdef *verdef)
        : PointerIntPair<const void*, 1>(verdef, 0) { }
    VersionMapEntry(const Elf_Vernaux *vernaux)
        : PointerIntPair<const void*, 1>(vernaux, 1) { }
    bool isNull() const { return getPointer() == NULL; }
    bool isVerdef() const { return !isNull() && getInt() == 0; }
    bool isVernaux() const { return !isNull() && getInt() == 1; }
    const Elf_Verdef *getVerdef() const {
      return isVerdef() ? (const Elf_Verdef*)getPointer() : NULL;
    }
    const Elf_Vernaux *getVernaux() const {
      return isVernaux() ? (const Elf_Vernaux*)getPointer() : NULL;
    }
  };
  mutable SmallVector<VersionMapEntry, 16> VersionMap;
  void LoadVersionDefs(const Elf_Shdr *sec) const;
  void LoadVersionNeeds(const Elf_Shdr *ec) const;
  void LoadVersionMap() const;

  /// @brief Map sections to an array of relocation sections that reference
  ///        them sorted by section index.
  RelocMap_t SectionRelocMap;

  /// @brief Get the relocation section that contains \a Rel.
  const Elf_Shdr *getRelSection(DataRefImpl Rel) const {
    return getSection(Rel.w.b);
  }

public:
  bool            isRelocationHasAddend(DataRefImpl Rel) const;
  template<typename T>
  const T        *getEntry(uint16_t Section, uint32_t Entry) const;
  template<typename T>
  const T        *getEntry(const Elf_Shdr *Section, uint32_t Entry) const;
  const Elf_Shdr *getSection(DataRefImpl index) const;
  const Elf_Shdr *getSection(uint32_t index) const;
  const Elf_Rel  *getRel(DataRefImpl Rel) const;
  const Elf_Rela *getRela(DataRefImpl Rela) const;
  const char     *getString(uint32_t section, uint32_t offset) const;
  const char     *getString(const Elf_Shdr *section, uint32_t offset) const;
  error_code      getSymbolVersion(const Elf_Shdr *section,
                                   const Elf_Sym *Symb,
                                   StringRef &Version,
                                   bool &IsDefault) const;
  void VerifyStrTab(const Elf_Shdr *sh) const;

protected:
  const Elf_Sym  *getSymbol(DataRefImpl Symb) const; // FIXME: Should be private?
  void            validateSymbol(DataRefImpl Symb) const;
  StringRef       getRelocationTypeName(uint32_t Type) const;

public:
  error_code      getSymbolName(const Elf_Shdr *section,
                                const Elf_Sym *Symb,
                                StringRef &Res) const;
  error_code      getSectionName(const Elf_Shdr *section,
                                 StringRef &Res) const;
  const Elf_Dyn  *getDyn(DataRefImpl DynData) const;
  error_code getSymbolVersion(SymbolRef Symb, StringRef &Version,
                              bool &IsDefault) const;
  uint64_t getSymbolIndex(const Elf_Sym *sym) const;
protected:
  virtual error_code getSymbolNext(DataRefImpl Symb, SymbolRef &Res) const;
  virtual error_code getSymbolName(DataRefImpl Symb, StringRef &Res) const;
  virtual error_code getSymbolFileOffset(DataRefImpl Symb, uint64_t &Res) const;
  virtual error_code getSymbolAddress(DataRefImpl Symb, uint64_t &Res) const;
  virtual error_code getSymbolAlignment(DataRefImpl Symb, uint32_t &Res) const;
  virtual error_code getSymbolSize(DataRefImpl Symb, uint64_t &Res) const;
  virtual error_code getSymbolNMTypeChar(DataRefImpl Symb, char &Res) const;
  virtual error_code getSymbolFlags(DataRefImpl Symb, uint32_t &Res) const;
  virtual error_code getSymbolType(DataRefImpl Symb, SymbolRef::Type &Res) const;
  virtual error_code getSymbolSection(DataRefImpl Symb,
                                      section_iterator &Res) const;
  virtual error_code getSymbolValue(DataRefImpl Symb, uint64_t &Val) const;

  virtual error_code getLibraryNext(DataRefImpl Data, LibraryRef &Result) const;
  virtual error_code getLibraryPath(DataRefImpl Data, StringRef &Res) const;

  virtual error_code getSectionNext(DataRefImpl Sec, SectionRef &Res) const;
  virtual error_code getSectionName(DataRefImpl Sec, StringRef &Res) const;
  virtual error_code getSectionAddress(DataRefImpl Sec, uint64_t &Res) const;
  virtual error_code getSectionSize(DataRefImpl Sec, uint64_t &Res) const;
  virtual error_code getSectionContents(DataRefImpl Sec, StringRef &Res) const;
  virtual error_code getSectionAlignment(DataRefImpl Sec, uint64_t &Res) const;
  virtual error_code isSectionText(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionData(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionBSS(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionRequiredForExecution(DataRefImpl Sec,
                                                   bool &Res) const;
  virtual error_code isSectionVirtual(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionZeroInit(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionReadOnlyData(DataRefImpl Sec, bool &Res) const;
  virtual error_code sectionContainsSymbol(DataRefImpl Sec, DataRefImpl Symb,
                                           bool &Result) const;
  virtual relocation_iterator getSectionRelBegin(DataRefImpl Sec) const;
  virtual relocation_iterator getSectionRelEnd(DataRefImpl Sec) const;

  virtual error_code getRelocationNext(DataRefImpl Rel,
                                       RelocationRef &Res) const;
  virtual error_code getRelocationAddress(DataRefImpl Rel,
                                          uint64_t &Res) const;
  virtual error_code getRelocationOffset(DataRefImpl Rel,
                                         uint64_t &Res) const;
  virtual error_code getRelocationSymbol(DataRefImpl Rel,
                                         SymbolRef &Res) const;
  virtual error_code getRelocationType(DataRefImpl Rel,
                                       uint64_t &Res) const;
  virtual error_code getRelocationTypeName(DataRefImpl Rel,
                                           SmallVectorImpl<char> &Result) const;
  virtual error_code getRelocationAdditionalInfo(DataRefImpl Rel,
                                                 int64_t &Res) const;
  virtual error_code getRelocationValueString(DataRefImpl Rel,
                                           SmallVectorImpl<char> &Result) const;

public:
  ELFObjectFile(MemoryBuffer *Object, error_code &ec);

  bool isMips64EL() const {
    return Header->e_machine == ELF::EM_MIPS &&
      Header->getFileClass() == ELF::ELFCLASS64 &&
      Header->getDataEncoding() == ELF::ELFDATA2LSB;
  }

  virtual symbol_iterator begin_symbols() const;
  virtual symbol_iterator end_symbols() const;

  virtual symbol_iterator begin_dynamic_symbols() const;
  virtual symbol_iterator end_dynamic_symbols() const;

  virtual section_iterator begin_sections() const;
  virtual section_iterator end_sections() const;

  virtual library_iterator begin_libraries_needed() const;
  virtual library_iterator end_libraries_needed() const;

  const Elf_Shdr *getDynamicSymbolTableSectionHeader() const {
    return SymbolTableSections[0];
  }

  const Elf_Shdr *getDynamicStringTableSectionHeader() const {
    return dot_dynstr_sec;
  }

  Elf_Dyn_iterator begin_dynamic_table() const;
  /// \param NULLEnd use one past the first DT_NULL entry as the end instead of
  /// the section size.
  Elf_Dyn_iterator end_dynamic_table(bool NULLEnd = false) const;

  Elf_Sym_iterator begin_elf_dynamic_symbols() const {
    const Elf_Shdr *DynSymtab = SymbolTableSections[0];
    if (DynSymtab)
      return Elf_Sym_iterator(DynSymtab->sh_entsize,
                              (const char *)base() + DynSymtab->sh_offset);
    return Elf_Sym_iterator(0, 0);
  }

  Elf_Sym_iterator end_elf_dynamic_symbols() const {
    const Elf_Shdr *DynSymtab = SymbolTableSections[0];
    if (DynSymtab)
      return Elf_Sym_iterator(DynSymtab->sh_entsize, (const char *)base() +
                              DynSymtab->sh_offset + DynSymtab->sh_size);
    return Elf_Sym_iterator(0, 0);
  }

  Elf_Rela_Iter beginELFRela(const Elf_Shdr *sec) const {
    return Elf_Rela_Iter(sec->sh_entsize,
                         (const char *)(base() + sec->sh_offset));
  }

  Elf_Rela_Iter endELFRela(const Elf_Shdr *sec) const {
    return Elf_Rela_Iter(sec->sh_entsize, (const char *)
                         (base() + sec->sh_offset + sec->sh_size));
  }

  Elf_Rel_Iter beginELFRel(const Elf_Shdr *sec) const {
    return Elf_Rel_Iter(sec->sh_entsize,
                        (const char *)(base() + sec->sh_offset));
  }

  Elf_Rel_Iter endELFRel(const Elf_Shdr *sec) const {
    return Elf_Rel_Iter(sec->sh_entsize, (const char *)
                        (base() + sec->sh_offset + sec->sh_size));
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

  virtual uint8_t getBytesInAddress() const;
  virtual StringRef getFileFormatName() const;
  virtual StringRef getObjectType() const { return "ELF"; }
  virtual unsigned getArch() const;
  virtual StringRef getLoadName() const;
  virtual error_code getSectionContents(const Elf_Shdr *sec,
                                        StringRef &Res) const;

  uint64_t getNumSections() const;
  uint64_t getStringTableIndex() const;
  ELF::Elf64_Word getSymbolTableIndex(const Elf_Sym *symb) const;
  const Elf_Ehdr *getElfHeader() const;
  const Elf_Shdr *getSection(const Elf_Sym *symb) const;
  const Elf_Shdr *getElfSection(section_iterator &It) const;
  const Elf_Sym *getElfSymbol(symbol_iterator &It) const;
  const Elf_Sym *getElfSymbol(uint32_t index) const;

  // Methods for type inquiry through isa, cast, and dyn_cast
  bool isDyldType() const { return isDyldELFObject; }
  static inline bool classof(const Binary *v) {
    return v->getType() == getELFType(ELFT::TargetEndianness == support::little,
                                      ELFT::Is64Bits);
  }
};

// Iterate through the version definitions, and place each Elf_Verdef
// in the VersionMap according to its index.
template<class ELFT>
void ELFObjectFile<ELFT>::LoadVersionDefs(const Elf_Shdr *sec) const {
  unsigned vd_size = sec->sh_size; // Size of section in bytes
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
      VersionMap.resize(index+1);
    VersionMap[index] = VersionMapEntry(vd);
    p += vd->vd_next;
  }
}

// Iterate through the versions needed section, and place each Elf_Vernaux
// in the VersionMap according to its index.
template<class ELFT>
void ELFObjectFile<ELFT>::LoadVersionNeeds(const Elf_Shdr *sec) const {
  unsigned vn_size = sec->sh_size; // Size of section in bytes
  unsigned vn_count = sec->sh_info; // Number of Verneed entries
  const char *sec_start = (const char*)base() + sec->sh_offset;
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
        VersionMap.resize(index+1);
      VersionMap[index] = VersionMapEntry(vna);
      paux += vna->vna_next;
    }
    p += vn->vn_next;
  }
}

template<class ELFT>
void ELFObjectFile<ELFT>::LoadVersionMap() const {
  // If there is no dynamic symtab or version table, there is nothing to do.
  if (SymbolTableSections[0] == NULL || dot_gnu_version_sec == NULL)
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

template<class ELFT>
void ELFObjectFile<ELFT>::validateSymbol(DataRefImpl Symb) const {
#ifndef NDEBUG
  const Elf_Sym  *symb = getSymbol(Symb);
  const Elf_Shdr *SymbolTableSection = SymbolTableSections[Symb.d.b];
  // FIXME: We really need to do proper error handling in the case of an invalid
  //        input file. Because we don't use exceptions, I think we'll just pass
  //        an error object around.
  if (!(  symb
        && SymbolTableSection
        && symb >= (const Elf_Sym*)(base()
                   + SymbolTableSection->sh_offset)
        && symb <  (const Elf_Sym*)(base()
                   + SymbolTableSection->sh_offset
                   + SymbolTableSection->sh_size)))
    // FIXME: Proper error handling.
    report_fatal_error("Symb must point to a valid symbol!");
#endif
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getSymbolNext(DataRefImpl Symb,
                                              SymbolRef &Result) const {
  validateSymbol(Symb);
  const Elf_Shdr *SymbolTableSection = SymbolTableSections[Symb.d.b];

  ++Symb.d.a;
  // Check to see if we are at the end of this symbol table.
  if (Symb.d.a >= SymbolTableSection->getEntityCount()) {
    // We are at the end. If there are other symbol tables, jump to them.
    // If the symbol table is .dynsym, we are iterating dynamic symbols,
    // and there is only one table of these.
    if (Symb.d.b != 0) {
      ++Symb.d.b;
      Symb.d.a = 1; // The 0th symbol in ELF is fake.
    }
    // Otherwise return the terminator.
    if (Symb.d.b == 0 || Symb.d.b >= SymbolTableSections.size()) {
      Symb.d.a = std::numeric_limits<uint32_t>::max();
      Symb.d.b = std::numeric_limits<uint32_t>::max();
    }
  }

  Result = SymbolRef(Symb, this);
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getSymbolName(DataRefImpl Symb,
                                              StringRef &Result) const {
  validateSymbol(Symb);
  const Elf_Sym *symb = getSymbol(Symb);
  return getSymbolName(SymbolTableSections[Symb.d.b], symb, Result);
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getSymbolVersion(SymbolRef SymRef,
                                                 StringRef &Version,
                                                 bool &IsDefault) const {
  DataRefImpl Symb = SymRef.getRawDataRefImpl();
  validateSymbol(Symb);
  const Elf_Sym *symb = getSymbol(Symb);
  return getSymbolVersion(SymbolTableSections[Symb.d.b], symb,
                          Version, IsDefault);
}

template<class ELFT>
ELF::Elf64_Word ELFObjectFile<ELFT>
                             ::getSymbolTableIndex(const Elf_Sym *symb) const {
  if (symb->st_shndx == ELF::SHN_XINDEX)
    return ExtendedSymbolTable.lookup(symb);
  return symb->st_shndx;
}

template<class ELFT>
const typename ELFObjectFile<ELFT>::Elf_Shdr *
ELFObjectFile<ELFT>::getSection(const Elf_Sym *symb) const {
  if (symb->st_shndx == ELF::SHN_XINDEX)
    return getSection(ExtendedSymbolTable.lookup(symb));
  if (symb->st_shndx >= ELF::SHN_LORESERVE)
    return 0;
  return getSection(symb->st_shndx);
}

template<class ELFT>
const typename ELFObjectFile<ELFT>::Elf_Ehdr *
ELFObjectFile<ELFT>::getElfHeader() const {
  return Header;
}

template<class ELFT>
const typename ELFObjectFile<ELFT>::Elf_Shdr *
ELFObjectFile<ELFT>::getElfSection(section_iterator &It) const {
  llvm::object::DataRefImpl ShdrRef = It->getRawDataRefImpl();
  return reinterpret_cast<const Elf_Shdr *>(ShdrRef.p);
}

template<class ELFT>
const typename ELFObjectFile<ELFT>::Elf_Sym *
ELFObjectFile<ELFT>::getElfSymbol(symbol_iterator &It) const {
  return getSymbol(It->getRawDataRefImpl());
}

template<class ELFT>
const typename ELFObjectFile<ELFT>::Elf_Sym *
ELFObjectFile<ELFT>::getElfSymbol(uint32_t index) const {
  DataRefImpl SymbolData;
  SymbolData.d.a = index;
  SymbolData.d.b = 1;
  return getSymbol(SymbolData);
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getSymbolFileOffset(DataRefImpl Symb,
                                                    uint64_t &Result) const {
  validateSymbol(Symb);
  const Elf_Sym  *symb = getSymbol(Symb);
  const Elf_Shdr *Section;
  switch (getSymbolTableIndex(symb)) {
  case ELF::SHN_COMMON:
   // Unintialized symbols have no offset in the object file
  case ELF::SHN_UNDEF:
    Result = UnknownAddressOrSize;
    return object_error::success;
  case ELF::SHN_ABS:
    Result = symb->st_value;
    return object_error::success;
  default: Section = getSection(symb);
  }

  switch (symb->getType()) {
  case ELF::STT_SECTION:
    Result = Section ? Section->sh_offset : UnknownAddressOrSize;
    return object_error::success;
  case ELF::STT_FUNC:
  case ELF::STT_OBJECT:
  case ELF::STT_NOTYPE:
    Result = symb->st_value +
             (Section ? Section->sh_offset : 0);
    return object_error::success;
  default:
    Result = UnknownAddressOrSize;
    return object_error::success;
  }
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getSymbolAddress(DataRefImpl Symb,
                                                 uint64_t &Result) const {
  validateSymbol(Symb);
  const Elf_Sym  *symb = getSymbol(Symb);
  const Elf_Shdr *Section;
  switch (getSymbolTableIndex(symb)) {
  case ELF::SHN_COMMON:
  case ELF::SHN_UNDEF:
    Result = UnknownAddressOrSize;
    return object_error::success;
  case ELF::SHN_ABS:
    Result = symb->st_value;
    return object_error::success;
  default: Section = getSection(symb);
  }

  switch (symb->getType()) {
  case ELF::STT_SECTION:
    Result = Section ? Section->sh_addr : UnknownAddressOrSize;
    return object_error::success;
  case ELF::STT_FUNC:
  case ELF::STT_OBJECT:
  case ELF::STT_NOTYPE:
    bool IsRelocatable;
    switch(Header->e_type) {
    case ELF::ET_EXEC:
    case ELF::ET_DYN:
      IsRelocatable = false;
      break;
    default:
      IsRelocatable = true;
    }
    Result = symb->st_value;

    // Clear the ARM/Thumb indicator flag.
    if (Header->e_machine == ELF::EM_ARM)
      Result &= ~1;

    if (IsRelocatable && Section != 0)
      Result += Section->sh_addr;
    return object_error::success;
  default:
    Result = UnknownAddressOrSize;
    return object_error::success;
  }
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getSymbolAlignment(DataRefImpl Symb,
                                                   uint32_t &Res) const {
  uint32_t flags;
  getSymbolFlags(Symb, flags);
  if (flags & SymbolRef::SF_Common) {
    uint64_t Value;
    getSymbolValue(Symb, Value);
    Res = Value;
  } else {
    Res = 0;
  }
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getSymbolSize(DataRefImpl Symb,
                                              uint64_t &Result) const {
  validateSymbol(Symb);
  const Elf_Sym  *symb = getSymbol(Symb);
  if (symb->st_size == 0)
    Result = UnknownAddressOrSize;
  Result = symb->st_size;
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getSymbolNMTypeChar(DataRefImpl Symb,
                                                    char &Result) const {
  validateSymbol(Symb);
  const Elf_Sym  *symb = getSymbol(Symb);
  const Elf_Shdr *Section = getSection(symb);

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

  switch (getSymbolTableIndex(symb)) {
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
    if (getSymbolTableIndex(symb) == ELF::SHN_UNDEF)
      ret = 'w';
    else
      if (symb->getType() == ELF::STT_OBJECT)
        ret = 'V';
      else
        ret = 'W';
  }

  if (ret == '?' && symb->getType() == ELF::STT_SECTION) {
    StringRef name;
    if (error_code ec = getSymbolName(Symb, name))
      return ec;
    Result = StringSwitch<char>(name)
      .StartsWith(".debug", 'N')
      .StartsWith(".note", 'n')
      .Default('?');
    return object_error::success;
  }

  Result = ret;
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getSymbolType(DataRefImpl Symb,
                                              SymbolRef::Type &Result) const {
  validateSymbol(Symb);
  const Elf_Sym  *symb = getSymbol(Symb);

  switch (symb->getType()) {
  case ELF::STT_NOTYPE:
    Result = SymbolRef::ST_Unknown;
    break;
  case ELF::STT_SECTION:
    Result = SymbolRef::ST_Debug;
    break;
  case ELF::STT_FILE:
    Result = SymbolRef::ST_File;
    break;
  case ELF::STT_FUNC:
    Result = SymbolRef::ST_Function;
    break;
  case ELF::STT_OBJECT:
  case ELF::STT_COMMON:
  case ELF::STT_TLS:
    Result = SymbolRef::ST_Data;
    break;
  default:
    Result = SymbolRef::ST_Other;
    break;
  }
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getSymbolFlags(DataRefImpl Symb,
                                               uint32_t &Result) const {
  validateSymbol(Symb);
  const Elf_Sym  *symb = getSymbol(Symb);

  Result = SymbolRef::SF_None;

  if (symb->getBinding() != ELF::STB_LOCAL)
    Result |= SymbolRef::SF_Global;

  if (symb->getBinding() == ELF::STB_WEAK)
    Result |= SymbolRef::SF_Weak;

  if (symb->st_shndx == ELF::SHN_ABS)
    Result |= SymbolRef::SF_Absolute;

  if (symb->getType() == ELF::STT_FILE ||
      symb->getType() == ELF::STT_SECTION)
    Result |= SymbolRef::SF_FormatSpecific;

  if (getSymbolTableIndex(symb) == ELF::SHN_UNDEF)
    Result |= SymbolRef::SF_Undefined;

  if (symb->getType() == ELF::STT_COMMON ||
      getSymbolTableIndex(symb) == ELF::SHN_COMMON)
    Result |= SymbolRef::SF_Common;

  if (symb->getType() == ELF::STT_TLS)
    Result |= SymbolRef::SF_ThreadLocal;

  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getSymbolSection(DataRefImpl Symb,
                                                 section_iterator &Res) const {
  validateSymbol(Symb);
  const Elf_Sym  *symb = getSymbol(Symb);
  const Elf_Shdr *sec = getSection(symb);
  if (!sec)
    Res = end_sections();
  else {
    DataRefImpl Sec;
    Sec.p = reinterpret_cast<intptr_t>(sec);
    Res = section_iterator(SectionRef(Sec, this));
  }
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getSymbolValue(DataRefImpl Symb,
                                               uint64_t &Val) const {
  validateSymbol(Symb);
  const Elf_Sym *symb = getSymbol(Symb);
  Val = symb->st_value;
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getSectionNext(DataRefImpl Sec,
                                               SectionRef &Result) const {
  const uint8_t *sec = reinterpret_cast<const uint8_t *>(Sec.p);
  sec += Header->e_shentsize;
  Sec.p = reinterpret_cast<intptr_t>(sec);
  Result = SectionRef(Sec, this);
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getSectionName(DataRefImpl Sec,
                                               StringRef &Result) const {
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  Result = StringRef(getString(dot_shstrtab_sec, sec->sh_name));
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getSectionAddress(DataRefImpl Sec,
                                                  uint64_t &Result) const {
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  Result = sec->sh_addr;
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getSectionSize(DataRefImpl Sec,
                                               uint64_t &Result) const {
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  Result = sec->sh_size;
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getSectionContents(DataRefImpl Sec,
                                                   StringRef &Result) const {
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  const char *start = (const char*)base() + sec->sh_offset;
  Result = StringRef(start, sec->sh_size);
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getSectionContents(const Elf_Shdr *Sec,
                                                   StringRef &Result) const {
  const char *start = (const char*)base() + Sec->sh_offset;
  Result = StringRef(start, Sec->sh_size);
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getSectionAlignment(DataRefImpl Sec,
                                                    uint64_t &Result) const {
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  Result = sec->sh_addralign;
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::isSectionText(DataRefImpl Sec,
                                              bool &Result) const {
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  if (sec->sh_flags & ELF::SHF_EXECINSTR)
    Result = true;
  else
    Result = false;
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::isSectionData(DataRefImpl Sec,
                                              bool &Result) const {
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  if (sec->sh_flags & (ELF::SHF_ALLOC | ELF::SHF_WRITE)
      && sec->sh_type == ELF::SHT_PROGBITS)
    Result = true;
  else
    Result = false;
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::isSectionBSS(DataRefImpl Sec,
                                             bool &Result) const {
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  if (sec->sh_flags & (ELF::SHF_ALLOC | ELF::SHF_WRITE)
      && sec->sh_type == ELF::SHT_NOBITS)
    Result = true;
  else
    Result = false;
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::isSectionRequiredForExecution(
    DataRefImpl Sec, bool &Result) const {
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  if (sec->sh_flags & ELF::SHF_ALLOC)
    Result = true;
  else
    Result = false;
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::isSectionVirtual(DataRefImpl Sec,
                                                 bool &Result) const {
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  if (sec->sh_type == ELF::SHT_NOBITS)
    Result = true;
  else
    Result = false;
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::isSectionZeroInit(DataRefImpl Sec,
                                                  bool &Result) const {
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  // For ELF, all zero-init sections are virtual (that is, they occupy no space
  //   in the object image) and vice versa.
  Result = sec->sh_type == ELF::SHT_NOBITS;
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::isSectionReadOnlyData(DataRefImpl Sec,
                                                      bool &Result) const {
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  if (sec->sh_flags & ELF::SHF_WRITE || sec->sh_flags & ELF::SHF_EXECINSTR)
    Result = false;
  else
    Result = true;
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::sectionContainsSymbol(DataRefImpl Sec,
                                                      DataRefImpl Symb,
                                                      bool &Result) const {
  validateSymbol(Symb);

  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  const Elf_Sym  *symb = getSymbol(Symb);

  unsigned shndx = symb->st_shndx;
  bool Reserved = shndx >= ELF::SHN_LORESERVE
               && shndx <= ELF::SHN_HIRESERVE;

  Result = !Reserved && (sec == getSection(symb->st_shndx));
  return object_error::success;
}

template<class ELFT>
relocation_iterator
ELFObjectFile<ELFT>::getSectionRelBegin(DataRefImpl Sec) const {
  DataRefImpl RelData;
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  typename RelocMap_t::const_iterator ittr = SectionRelocMap.find(sec);
  if (sec != 0 && ittr != SectionRelocMap.end()) {
    RelData.w.a = getSection(ittr->second[0])->sh_info;
    RelData.w.b = ittr->second[0];
    RelData.w.c = 0;
  }
  return relocation_iterator(RelocationRef(RelData, this));
}

template<class ELFT>
relocation_iterator
ELFObjectFile<ELFT>::getSectionRelEnd(DataRefImpl Sec) const {
  DataRefImpl RelData;
  const Elf_Shdr *sec = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  typename RelocMap_t::const_iterator ittr = SectionRelocMap.find(sec);
  if (sec != 0 && ittr != SectionRelocMap.end()) {
    // Get the index of the last relocation section for this section.
    std::size_t relocsecindex = ittr->second[ittr->second.size() - 1];
    const Elf_Shdr *relocsec = getSection(relocsecindex);
    RelData.w.a = relocsec->sh_info;
    RelData.w.b = relocsecindex;
    RelData.w.c = relocsec->sh_size / relocsec->sh_entsize;
  }
  return relocation_iterator(RelocationRef(RelData, this));
}

// Relocations
template<class ELFT>
error_code ELFObjectFile<ELFT>::getRelocationNext(DataRefImpl Rel,
                                                  RelocationRef &Result) const {
  ++Rel.w.c;
  const Elf_Shdr *relocsec = getSection(Rel.w.b);
  if (Rel.w.c >= (relocsec->sh_size / relocsec->sh_entsize)) {
    // We have reached the end of the relocations for this section. See if there
    // is another relocation section.
    typename RelocMap_t::mapped_type relocseclist =
      SectionRelocMap.lookup(getSection(Rel.w.a));

    // Do a binary search for the current reloc section index (which must be
    // present). Then get the next one.
    typename RelocMap_t::mapped_type::const_iterator loc =
      std::lower_bound(relocseclist.begin(), relocseclist.end(), Rel.w.b);
    ++loc;

    // If there is no next one, don't do anything. The ++Rel.w.c above sets Rel
    // to the end iterator.
    if (loc != relocseclist.end()) {
      Rel.w.b = *loc;
      Rel.w.a = 0;
    }
  }
  Result = RelocationRef(Rel, this);
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getRelocationSymbol(DataRefImpl Rel,
                                                    SymbolRef &Result) const {
  uint32_t symbolIdx;
  const Elf_Shdr *sec = getSection(Rel.w.b);
  switch (sec->sh_type) {
    default :
      report_fatal_error("Invalid section type in Rel!");
    case ELF::SHT_REL : {
      symbolIdx = getRel(Rel)->getSymbol(isMips64EL());
      break;
    }
    case ELF::SHT_RELA : {
      symbolIdx = getRela(Rel)->getSymbol(isMips64EL());
      break;
    }
  }
  DataRefImpl SymbolData;
  IndexMap_t::const_iterator it = SymbolTableSectionsIndexMap.find(sec->sh_link);
  if (it == SymbolTableSectionsIndexMap.end())
    report_fatal_error("Relocation symbol table not found!");
  SymbolData.d.a = symbolIdx;
  SymbolData.d.b = it->second;
  Result = SymbolRef(SymbolData, this);
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getRelocationAddress(DataRefImpl Rel,
                                                     uint64_t &Result) const {
  assert((Header->e_type == ELF::ET_EXEC || Header->e_type == ELF::ET_DYN) &&
         "Only executable and shared objects files have addresses");
  Result = getROffset(Rel);
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getRelocationOffset(DataRefImpl Rel,
                                                    uint64_t &Result) const {
  assert(Header->e_type == ELF::ET_REL &&
         "Only relocatable object files have relocation offsets");
  Result = getROffset(Rel);
  return object_error::success;
}

template<class ELFT>
uint64_t ELFObjectFile<ELFT>::getROffset(DataRefImpl Rel) const {
  const Elf_Shdr *sec = getSection(Rel.w.b);
  switch (sec->sh_type) {
  default:
    report_fatal_error("Invalid section type in Rel!");
  case ELF::SHT_REL:
    return getRel(Rel)->r_offset;
  case ELF::SHT_RELA:
    return getRela(Rel)->r_offset;
  }
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getRelocationType(DataRefImpl Rel,
                                                  uint64_t &Result) const {
  const Elf_Shdr *sec = getSection(Rel.w.b);
  switch (sec->sh_type) {
    default :
      report_fatal_error("Invalid section type in Rel!");
    case ELF::SHT_REL : {
      Result = getRel(Rel)->getType(isMips64EL());
      break;
    }
    case ELF::SHT_RELA : {
      Result = getRela(Rel)->getType(isMips64EL());
      break;
    }
  }
  return object_error::success;
}

#define LLVM_ELF_SWITCH_RELOC_TYPE_NAME(enum) \
  case ELF::enum: Res = #enum; break;

template<class ELFT>
StringRef ELFObjectFile<ELFT>::getRelocationTypeName(uint32_t Type) const {
  StringRef Res = "Unknown";
  switch (Header->e_machine) {
  case ELF::EM_X86_64:
    switch (Type) {
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_NONE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_PC32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_GOT32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_PLT32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_COPY);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_GLOB_DAT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_JUMP_SLOT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_RELATIVE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_GOTPCREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_32S);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_PC16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_8);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_PC8);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_DTPMOD64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_DTPOFF64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_TPOFF64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_TLSGD);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_TLSLD);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_DTPOFF32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_GOTTPOFF);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_TPOFF32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_PC64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_GOTOFF64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_GOTPC32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_GOT64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_GOTPCREL64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_GOTPC64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_GOTPLT64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_PLTOFF64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_SIZE32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_SIZE64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_GOTPC32_TLSDESC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_TLSDESC_CALL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_TLSDESC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_X86_64_IRELATIVE);
    default: break;
    }
    break;
  case ELF::EM_386:
    switch (Type) {
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_NONE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_PC32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_GOT32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_PLT32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_COPY);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_GLOB_DAT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_JUMP_SLOT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_RELATIVE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_GOTOFF);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_GOTPC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_32PLT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_TPOFF);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_IE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_GOTIE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_LE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_GD);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_LDM);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_PC16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_8);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_PC8);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_GD_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_GD_PUSH);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_GD_CALL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_GD_POP);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_LDM_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_LDM_PUSH);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_LDM_CALL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_LDM_POP);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_LDO_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_IE_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_LE_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_DTPMOD32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_DTPOFF32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_TPOFF32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_GOTDESC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_DESC_CALL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_TLS_DESC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_386_IRELATIVE);
    default: break;
    }
    break;
  case ELF::EM_MIPS:
    switch (Type) {
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_NONE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_REL32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_26);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_HI16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_LO16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_GPREL16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_LITERAL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_GOT16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_PC16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_CALL16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_GPREL32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_SHIFT5);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_SHIFT6);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_GOT_DISP);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_GOT_PAGE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_GOT_OFST);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_GOT_HI16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_GOT_LO16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_SUB);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_INSERT_A);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_INSERT_B);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_DELETE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_HIGHER);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_HIGHEST);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_CALL_HI16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_CALL_LO16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_SCN_DISP);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_REL16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_ADD_IMMEDIATE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_PJUMP);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_RELGOT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_JALR);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_TLS_DTPMOD32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_TLS_DTPREL32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_TLS_DTPMOD64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_TLS_DTPREL64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_TLS_GD);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_TLS_LDM);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_TLS_DTPREL_HI16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_TLS_DTPREL_LO16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_TLS_GOTTPREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_TLS_TPREL32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_TLS_TPREL64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_TLS_TPREL_HI16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_TLS_TPREL_LO16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_GLOB_DAT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_COPY);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_JUMP_SLOT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_MIPS_NUM);
    default: break;
    }
    break;
  case ELF::EM_AARCH64:
    switch (Type) {
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_NONE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_ABS64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_ABS32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_ABS16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_PREL64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_PREL32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_PREL16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_MOVW_UABS_G0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_MOVW_UABS_G0_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_MOVW_UABS_G1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_MOVW_UABS_G1_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_MOVW_UABS_G2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_MOVW_UABS_G2_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_MOVW_UABS_G3);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_MOVW_SABS_G0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_MOVW_SABS_G1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_MOVW_SABS_G2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_LD_PREL_LO19);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_ADR_PREL_LO21);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_ADR_PREL_PG_HI21);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_ADD_ABS_LO12_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_LDST8_ABS_LO12_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TSTBR14);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_CONDBR19);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_JUMP26);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_CALL26);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_LDST16_ABS_LO12_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_LDST32_ABS_LO12_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_LDST64_ABS_LO12_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_LDST128_ABS_LO12_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_ADR_GOT_PAGE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_LD64_GOT_LO12_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLD_MOVW_DTPREL_G2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLD_MOVW_DTPREL_G1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLD_MOVW_DTPREL_G1_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLD_MOVW_DTPREL_G0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLD_MOVW_DTPREL_G0_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLD_ADD_DTPREL_HI12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLD_ADD_DTPREL_LO12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLD_ADD_DTPREL_LO12_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLD_LDST8_DTPREL_LO12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLD_LDST8_DTPREL_LO12_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLD_LDST16_DTPREL_LO12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLD_LDST16_DTPREL_LO12_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLD_LDST32_DTPREL_LO12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLD_LDST32_DTPREL_LO12_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLD_LDST64_DTPREL_LO12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLD_LDST64_DTPREL_LO12_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSIE_MOVW_GOTTPREL_G1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSIE_MOVW_GOTTPREL_G0_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSIE_LD_GOTTPREL_PREL19);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLE_MOVW_TPREL_G2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLE_MOVW_TPREL_G1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLE_MOVW_TPREL_G1_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLE_MOVW_TPREL_G0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLE_MOVW_TPREL_G0_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLE_ADD_TPREL_HI12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLE_ADD_TPREL_LO12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLE_ADD_TPREL_LO12_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLE_LDST8_TPREL_LO12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLE_LDST8_TPREL_LO12_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLE_LDST16_TPREL_LO12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLE_LDST16_TPREL_LO12_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLE_LDST32_TPREL_LO12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLE_LDST32_TPREL_LO12_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLE_LDST64_TPREL_LO12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSLE_LDST64_TPREL_LO12_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSDESC_ADR_PAGE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSDESC_LD64_LO12_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSDESC_ADD_LO12_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_AARCH64_TLSDESC_CALL);
    default: break;
    }
    break;
  case ELF::EM_ARM:
    switch (Type) {
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_NONE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PC24);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ABS32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_REL32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDR_PC_G0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ABS16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ABS12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_ABS5);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ABS8);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_SBREL32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_CALL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_PC8);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_BREL_ADJ);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_DESC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_SWI8);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_XPC25);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_XPC22);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_DTPMOD32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_DTPOFF32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_TPOFF32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_COPY);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_GLOB_DAT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_JUMP_SLOT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_RELATIVE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_GOTOFF32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_BASE_PREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_GOT_BREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PLT32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_CALL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_JUMP24);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_JUMP24);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_BASE_ABS);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_PCREL_7_0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_PCREL_15_8);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_PCREL_23_15);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDR_SBREL_11_0_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_SBREL_19_12_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_SBREL_27_20_CK);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TARGET1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_SBREL31);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_V4BX);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TARGET2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PREL31);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_MOVW_ABS_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_MOVT_ABS);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_MOVW_PREL_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_MOVT_PREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_MOVW_ABS_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_MOVT_ABS);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_MOVW_PREL_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_MOVT_PREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_JUMP19);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_JUMP6);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_ALU_PREL_11_0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_PC12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ABS32_NOI);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_REL32_NOI);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_PC_G0_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_PC_G0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_PC_G1_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_PC_G1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_PC_G2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDR_PC_G1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDR_PC_G2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDRS_PC_G0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDRS_PC_G1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDRS_PC_G2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDC_PC_G0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDC_PC_G1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDC_PC_G2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_SB_G0_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_SB_G0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_SB_G1_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_SB_G1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ALU_SB_G2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDR_SB_G0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDR_SB_G1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDR_SB_G2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDRS_SB_G0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDRS_SB_G1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDRS_SB_G2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDC_SB_G0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDC_SB_G1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_LDC_SB_G2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_MOVW_BREL_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_MOVT_BREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_MOVW_BREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_MOVW_BREL_NC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_MOVT_BREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_MOVW_BREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_GOTDESC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_CALL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_DESCSEQ);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_TLS_CALL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PLT32_ABS);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_GOT_ABS);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_GOT_PREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_GOT_BREL12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_GOTOFF12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_GOTRELAX);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_GNU_VTENTRY);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_GNU_VTINHERIT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_JUMP11);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_JUMP8);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_GD32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_LDM32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_LDO32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_IE32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_LE32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_LDO12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_LE12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_TLS_IE12GP);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_3);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_4);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_5);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_6);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_7);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_8);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_9);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_10);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_11);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_13);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_14);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_PRIVATE_15);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_ME_TOO);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_TLS_DESCSEQ16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_ARM_THM_TLS_DESCSEQ32);
    default: break;
    }
    break;
  case ELF::EM_HEXAGON:
    switch (Type) {
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_NONE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_B22_PCREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_B15_PCREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_B7_PCREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_LO16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_HI16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_8);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GPREL16_0);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GPREL16_1);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GPREL16_2);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GPREL16_3);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_HL16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_B13_PCREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_B9_PCREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_B32_PCREL_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_32_6_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_B22_PCREL_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_B15_PCREL_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_B13_PCREL_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_B9_PCREL_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_B7_PCREL_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_16_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_12_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_11_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_10_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_9_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_8_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_7_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_6_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_32_PCREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_COPY);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GLOB_DAT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_JMP_SLOT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_RELATIVE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_PLT_B22_PCREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOTREL_LO16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOTREL_HI16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOTREL_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOT_LO16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOT_HI16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOT_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOT_16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_DTPMOD_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_DTPREL_LO16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_DTPREL_HI16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_DTPREL_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_DTPREL_16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GD_PLT_B22_PCREL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GD_GOT_LO16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GD_GOT_HI16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GD_GOT_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GD_GOT_16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_IE_LO16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_IE_HI16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_IE_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_IE_GOT_LO16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_IE_GOT_HI16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_IE_GOT_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_IE_GOT_16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_TPREL_LO16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_TPREL_HI16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_TPREL_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_TPREL_16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_6_PCREL_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOTREL_32_6_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOTREL_16_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOTREL_11_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOT_32_6_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOT_16_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GOT_11_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_DTPREL_32_6_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_DTPREL_16_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_DTPREL_11_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GD_GOT_32_6_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GD_GOT_16_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_GD_GOT_11_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_IE_32_6_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_IE_16_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_IE_GOT_32_6_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_IE_GOT_16_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_IE_GOT_11_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_TPREL_32_6_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_TPREL_16_X);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_HEX_TPREL_11_X);
    default: break;
    }
    break;
  case ELF::EM_PPC:
    switch (Type) {
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC_NONE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC_ADDR32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC_ADDR24);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC_ADDR16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC_ADDR16_LO);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC_ADDR16_HI);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC_ADDR16_HA);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC_ADDR14);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC_ADDR14_BRTAKEN);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC_ADDR14_BRNTAKEN);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC_REL24);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC_REL14);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC_REL14_BRTAKEN);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC_REL14_BRNTAKEN);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC_REL32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC_TPREL16_LO);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC_TPREL16_HA);
    default: break;
    }
    break;
  case ELF::EM_PPC64:
    switch (Type) {
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_NONE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_ADDR32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_ADDR16_LO);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_ADDR16_HI);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_ADDR14);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_REL24);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_REL32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_ADDR64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_ADDR16_HIGHER);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_ADDR16_HIGHEST);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_REL64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_TOC16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_TOC16_LO);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_TOC16_HA);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_TOC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_ADDR16_DS);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_ADDR16_LO_DS);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_TOC16_DS);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_TOC16_LO_DS);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_TLS);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_TPREL16_LO);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_TPREL16_HA);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_DTPREL16_LO);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_DTPREL16_HA);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_GOT_TLSGD16_LO);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_GOT_TLSGD16_HA);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_GOT_TLSLD16_LO);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_GOT_TLSLD16_HA);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_GOT_TPREL16_LO_DS);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_GOT_TPREL16_HA);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_TLSGD);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_PPC64_TLSLD);
    default: break;
    }
    break;
  case ELF::EM_S390:
    switch (Type) {
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_NONE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_8);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_PC32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_GOT12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_GOT32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_PLT32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_COPY);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_GLOB_DAT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_JMP_SLOT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_RELATIVE);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_GOTOFF);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_GOTPC);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_GOT16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_PC16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_PC16DBL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_PLT16DBL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_PC32DBL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_PLT32DBL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_GOTPCDBL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_PC64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_GOT64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_PLT64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_GOTENT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_GOTOFF16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_GOTOFF64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_GOTPLT12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_GOTPLT16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_GOTPLT32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_GOTPLT64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_GOTPLTENT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_PLTOFF16);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_PLTOFF32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_PLTOFF64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_TLS_LOAD);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_TLS_GDCALL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_TLS_LDCALL);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_TLS_GD32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_TLS_GD64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_TLS_GOTIE12);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_TLS_GOTIE32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_TLS_GOTIE64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_TLS_LDM32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_TLS_LDM64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_TLS_IE32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_TLS_IE64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_TLS_IEENT);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_TLS_LE32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_TLS_LE64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_TLS_LDO32);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_TLS_LDO64);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_TLS_DTPMOD);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_TLS_DTPOFF);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_TLS_TPOFF);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_20);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_GOT20);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_GOTPLT20);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_TLS_GOTIE20);
      LLVM_ELF_SWITCH_RELOC_TYPE_NAME(R_390_IRELATIVE);
    default: break;
    }
    break;
  default: break;
  }
  return Res;
}

#undef LLVM_ELF_SWITCH_RELOC_TYPE_NAME

template<class ELFT>
error_code ELFObjectFile<ELFT>::getRelocationTypeName(
    DataRefImpl Rel, SmallVectorImpl<char> &Result) const {
  const Elf_Shdr *sec = getSection(Rel.w.b);
  uint32_t type;
  switch (sec->sh_type) {
    default :
      return object_error::parse_failed;
    case ELF::SHT_REL : {
      type = getRel(Rel)->getType(isMips64EL());
      break;
    }
    case ELF::SHT_RELA : {
      type = getRela(Rel)->getType(isMips64EL());
      break;
    }
  }

  if (!isMips64EL()) {
    StringRef Name = getRelocationTypeName(type);
    Result.append(Name.begin(), Name.end());
  } else {
    uint8_t Type1 = (type >>  0) & 0xFF;
    uint8_t Type2 = (type >>  8) & 0xFF;
    uint8_t Type3 = (type >> 16) & 0xFF;

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

  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getRelocationAdditionalInfo(
    DataRefImpl Rel, int64_t &Result) const {
  const Elf_Shdr *sec = getSection(Rel.w.b);
  switch (sec->sh_type) {
    default :
      report_fatal_error("Invalid section type in Rel!");
    case ELF::SHT_REL : {
      Result = 0;
      return object_error::success;
    }
    case ELF::SHT_RELA : {
      Result = getRela(Rel)->r_addend;
      return object_error::success;
    }
  }
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getRelocationValueString(
    DataRefImpl Rel, SmallVectorImpl<char> &Result) const {
  const Elf_Shdr *sec = getSection(Rel.w.b);
  uint8_t type;
  StringRef res;
  int64_t addend = 0;
  uint16_t symbol_index = 0;
  switch (sec->sh_type) {
    default:
      return object_error::parse_failed;
    case ELF::SHT_REL: {
      type = getRel(Rel)->getType(isMips64EL());
      symbol_index = getRel(Rel)->getSymbol(isMips64EL());
      // TODO: Read implicit addend from section data.
      break;
    }
    case ELF::SHT_RELA: {
      type = getRela(Rel)->getType(isMips64EL());
      symbol_index = getRela(Rel)->getSymbol(isMips64EL());
      addend = getRela(Rel)->r_addend;
      break;
    }
  }
  const Elf_Sym *symb = getEntry<Elf_Sym>(sec->sh_link, symbol_index);
  StringRef symname;
  if (error_code ec = getSymbolName(getSection(sec->sh_link), symb, symname))
    return ec;
  switch (Header->e_machine) {
  case ELF::EM_X86_64:
    switch (type) {
    case ELF::R_X86_64_PC8:
    case ELF::R_X86_64_PC16:
    case ELF::R_X86_64_PC32: {
        std::string fmtbuf;
        raw_string_ostream fmt(fmtbuf);
        fmt << symname << (addend < 0 ? "" : "+") << addend << "-P";
        fmt.flush();
        Result.append(fmtbuf.begin(), fmtbuf.end());
      }
      break;
    case ELF::R_X86_64_8:
    case ELF::R_X86_64_16:
    case ELF::R_X86_64_32:
    case ELF::R_X86_64_32S:
    case ELF::R_X86_64_64: {
        std::string fmtbuf;
        raw_string_ostream fmt(fmtbuf);
        fmt << symname << (addend < 0 ? "" : "+") << addend;
        fmt.flush();
        Result.append(fmtbuf.begin(), fmtbuf.end());
      }
      break;
    default:
      res = "Unknown";
    }
    break;
  case ELF::EM_AARCH64:
  case ELF::EM_ARM:
  case ELF::EM_HEXAGON:
    res = symname;
    break;
  default:
    res = "Unknown";
  }
  if (Result.empty())
    Result.append(res.begin(), res.end());
  return object_error::success;
}

// Verify that the last byte in the string table in a null.
template<class ELFT>
void ELFObjectFile<ELFT>::VerifyStrTab(const Elf_Shdr *sh) const {
  const char *strtab = (const char*)base() + sh->sh_offset;
  if (strtab[sh->sh_size - 1] != 0)
    // FIXME: Proper error handling.
    report_fatal_error("String table must end with a null terminator!");
}

template<class ELFT>
ELFObjectFile<ELFT>::ELFObjectFile(MemoryBuffer *Object, error_code &ec)
  : ObjectFile(getELFType(
      static_cast<endianness>(ELFT::TargetEndianness) == support::little,
      ELFT::Is64Bits),
      Object)
  , isDyldELFObject(false)
  , SectionHeaderTable(0)
  , dot_shstrtab_sec(0)
  , dot_strtab_sec(0)
  , dot_dynstr_sec(0)
  , dot_dynamic_sec(0)
  , dot_gnu_version_sec(0)
  , dot_gnu_version_r_sec(0)
  , dot_gnu_version_d_sec(0)
  , dt_soname(0)
 {

  const uint64_t FileSize = Data->getBufferSize();

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

  // To find the symbol tables we walk the section table to find SHT_SYMTAB.
  const Elf_Shdr* SymbolTableSectionHeaderIndex = 0;
  const Elf_Shdr* sh = SectionHeaderTable;

  // Reserve SymbolTableSections[0] for .dynsym
  SymbolTableSections.push_back(NULL);

  for (uint64_t i = 0, e = getNumSections(); i != e; ++i) {
    switch (sh->sh_type) {
    case ELF::SHT_SYMTAB_SHNDX: {
      if (SymbolTableSectionHeaderIndex)
        // FIXME: Proper error handling.
        report_fatal_error("More than one .symtab_shndx!");
      SymbolTableSectionHeaderIndex = sh;
      break;
    }
    case ELF::SHT_SYMTAB: {
      SymbolTableSectionsIndexMap[i] = SymbolTableSections.size();
      SymbolTableSections.push_back(sh);
      break;
    }
    case ELF::SHT_DYNSYM: {
      if (SymbolTableSections[0] != NULL)
        // FIXME: Proper error handling.
        report_fatal_error("More than one .dynsym!");
      SymbolTableSectionsIndexMap[i] = 0;
      SymbolTableSections[0] = sh;
      break;
    }
    case ELF::SHT_REL:
    case ELF::SHT_RELA: {
      SectionRelocMap[getSection(sh->sh_info)].push_back(i);
      break;
    }
    case ELF::SHT_DYNAMIC: {
      if (dot_dynamic_sec != NULL)
        // FIXME: Proper error handling.
        report_fatal_error("More than one .dynamic!");
      dot_dynamic_sec = sh;
      break;
    }
    case ELF::SHT_GNU_versym: {
      if (dot_gnu_version_sec != NULL)
        // FIXME: Proper error handling.
        report_fatal_error("More than one .gnu.version section!");
      dot_gnu_version_sec = sh;
      break;
    }
    case ELF::SHT_GNU_verdef: {
      if (dot_gnu_version_d_sec != NULL)
        // FIXME: Proper error handling.
        report_fatal_error("More than one .gnu.version_d section!");
      dot_gnu_version_d_sec = sh;
      break;
    }
    case ELF::SHT_GNU_verneed: {
      if (dot_gnu_version_r_sec != NULL)
        // FIXME: Proper error handling.
        report_fatal_error("More than one .gnu.version_r section!");
      dot_gnu_version_r_sec = sh;
      break;
    }
    }
    ++sh;
  }

  // Sort section relocation lists by index.
  for (typename RelocMap_t::iterator i = SectionRelocMap.begin(),
                                     e = SectionRelocMap.end(); i != e; ++i) {
    std::sort(i->second.begin(), i->second.end());
  }

  // Get string table sections.
  dot_shstrtab_sec = getSection(getStringTableIndex());
  if (dot_shstrtab_sec) {
    // Verify that the last byte in the string table in a null.
    VerifyStrTab(dot_shstrtab_sec);
  }

  // Merge this into the above loop.
  for (const char *i = reinterpret_cast<const char *>(SectionHeaderTable),
                  *e = i + getNumSections() * Header->e_shentsize;
                   i != e; i += Header->e_shentsize) {
    const Elf_Shdr *sh = reinterpret_cast<const Elf_Shdr*>(i);
    if (sh->sh_type == ELF::SHT_STRTAB) {
      StringRef SectionName(getString(dot_shstrtab_sec, sh->sh_name));
      if (SectionName == ".strtab") {
        if (dot_strtab_sec != 0)
          // FIXME: Proper error handling.
          report_fatal_error("Already found section named .strtab!");
        dot_strtab_sec = sh;
        VerifyStrTab(dot_strtab_sec);
      } else if (SectionName == ".dynstr") {
        if (dot_dynstr_sec != 0)
          // FIXME: Proper error handling.
          report_fatal_error("Already found section named .dynstr!");
        dot_dynstr_sec = sh;
        VerifyStrTab(dot_dynstr_sec);
      }
    }
  }

  // Build symbol name side-mapping if there is one.
  if (SymbolTableSectionHeaderIndex) {
    const Elf_Word *ShndxTable = reinterpret_cast<const Elf_Word*>(base() +
                                      SymbolTableSectionHeaderIndex->sh_offset);
    error_code ec;
    for (symbol_iterator si = begin_symbols(),
                         se = end_symbols(); si != se; si.increment(ec)) {
      if (ec)
        report_fatal_error("Fewer extended symbol table entries than symbols!");
      if (*ShndxTable != ELF::SHN_UNDEF)
        ExtendedSymbolTable[getSymbol(si->getRawDataRefImpl())] = *ShndxTable;
      ++ShndxTable;
    }
  }
}

// Get the symbol table index in the symtab section given a symbol
template<class ELFT>
uint64_t ELFObjectFile<ELFT>::getSymbolIndex(const Elf_Sym *Sym) const {
  assert(SymbolTableSections.size() == 1 && "Only one symbol table supported!");
  const Elf_Shdr *SymTab = *SymbolTableSections.begin();
  uintptr_t SymLoc = uintptr_t(Sym);
  uintptr_t SymTabLoc = uintptr_t(base() + SymTab->sh_offset);
  assert(SymLoc > SymTabLoc && "Symbol not in symbol table!");
  uint64_t SymOffset = SymLoc - SymTabLoc;
  assert(SymOffset % SymTab->sh_entsize == 0 &&
         "Symbol not multiple of symbol size!");
  return SymOffset / SymTab->sh_entsize;
}

template<class ELFT>
symbol_iterator ELFObjectFile<ELFT>::begin_symbols() const {
  DataRefImpl SymbolData;
  if (SymbolTableSections.size() <= 1) {
    SymbolData.d.a = std::numeric_limits<uint32_t>::max();
    SymbolData.d.b = std::numeric_limits<uint32_t>::max();
  } else {
    SymbolData.d.a = 1; // The 0th symbol in ELF is fake.
    SymbolData.d.b = 1; // The 0th table is .dynsym
  }
  return symbol_iterator(SymbolRef(SymbolData, this));
}

template<class ELFT>
symbol_iterator ELFObjectFile<ELFT>::end_symbols() const {
  DataRefImpl SymbolData;
  SymbolData.d.a = std::numeric_limits<uint32_t>::max();
  SymbolData.d.b = std::numeric_limits<uint32_t>::max();
  return symbol_iterator(SymbolRef(SymbolData, this));
}

template<class ELFT>
symbol_iterator ELFObjectFile<ELFT>::begin_dynamic_symbols() const {
  DataRefImpl SymbolData;
  if (SymbolTableSections[0] == NULL) {
    SymbolData.d.a = std::numeric_limits<uint32_t>::max();
    SymbolData.d.b = std::numeric_limits<uint32_t>::max();
  } else {
    SymbolData.d.a = 1; // The 0th symbol in ELF is fake.
    SymbolData.d.b = 0; // The 0th table is .dynsym
  }
  return symbol_iterator(SymbolRef(SymbolData, this));
}

template<class ELFT>
symbol_iterator ELFObjectFile<ELFT>::end_dynamic_symbols() const {
  DataRefImpl SymbolData;
  SymbolData.d.a = std::numeric_limits<uint32_t>::max();
  SymbolData.d.b = std::numeric_limits<uint32_t>::max();
  return symbol_iterator(SymbolRef(SymbolData, this));
}

template<class ELFT>
section_iterator ELFObjectFile<ELFT>::begin_sections() const {
  DataRefImpl ret;
  ret.p = reinterpret_cast<intptr_t>(base() + Header->e_shoff);
  return section_iterator(SectionRef(ret, this));
}

template<class ELFT>
section_iterator ELFObjectFile<ELFT>::end_sections() const {
  DataRefImpl ret;
  ret.p = reinterpret_cast<intptr_t>(base()
                                     + Header->e_shoff
                                     + (Header->e_shentsize*getNumSections()));
  return section_iterator(SectionRef(ret, this));
}

template<class ELFT>
typename ELFObjectFile<ELFT>::Elf_Dyn_iterator
ELFObjectFile<ELFT>::begin_dynamic_table() const {
  if (dot_dynamic_sec)
    return Elf_Dyn_iterator(dot_dynamic_sec->sh_entsize,
                            (const char *)base() + dot_dynamic_sec->sh_offset);
  return Elf_Dyn_iterator(0, 0);
}

template<class ELFT>
typename ELFObjectFile<ELFT>::Elf_Dyn_iterator
ELFObjectFile<ELFT>::end_dynamic_table(bool NULLEnd) const {
  if (dot_dynamic_sec) {
    Elf_Dyn_iterator Ret(dot_dynamic_sec->sh_entsize,
                         (const char *)base() + dot_dynamic_sec->sh_offset +
                         dot_dynamic_sec->sh_size);

    if (NULLEnd) {
      Elf_Dyn_iterator Start = begin_dynamic_table();
      while (Start != Ret && Start->getTag() != ELF::DT_NULL)
        ++Start;

      // Include the DT_NULL.
      if (Start != Ret)
        ++Start;
      Ret = Start;
    }
    return Ret;
  }
  return Elf_Dyn_iterator(0, 0);
}

template<class ELFT>
StringRef ELFObjectFile<ELFT>::getLoadName() const {
  if (!dt_soname) {
    // Find the DT_SONAME entry
    Elf_Dyn_iterator it = begin_dynamic_table();
    Elf_Dyn_iterator ie = end_dynamic_table();
    while (it != ie && it->getTag() != ELF::DT_SONAME)
      ++it;

    if (it != ie) {
      if (dot_dynstr_sec == NULL)
        report_fatal_error("Dynamic string table is missing");
      dt_soname = getString(dot_dynstr_sec, it->getVal());
    } else {
      dt_soname = "";
    }
  }
  return dt_soname;
}

template<class ELFT>
library_iterator ELFObjectFile<ELFT>::begin_libraries_needed() const {
  // Find the first DT_NEEDED entry
  Elf_Dyn_iterator i = begin_dynamic_table();
  Elf_Dyn_iterator e = end_dynamic_table();
  while (i != e && i->getTag() != ELF::DT_NEEDED)
    ++i;

  DataRefImpl DRI;
  DRI.p = reinterpret_cast<uintptr_t>(i.get());
  return library_iterator(LibraryRef(DRI, this));
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getLibraryNext(DataRefImpl Data,
                                               LibraryRef &Result) const {
  // Use the same DataRefImpl format as DynRef.
  Elf_Dyn_iterator i = Elf_Dyn_iterator(dot_dynamic_sec->sh_entsize,
                                        reinterpret_cast<const char *>(Data.p));
  Elf_Dyn_iterator e = end_dynamic_table();

  // Skip the current dynamic table entry and find the next DT_NEEDED entry.
  do
    ++i;
  while (i != e && i->getTag() != ELF::DT_NEEDED);

  DataRefImpl DRI;
  DRI.p = reinterpret_cast<uintptr_t>(i.get());
  Result = LibraryRef(DRI, this);
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getLibraryPath(DataRefImpl Data,
                                               StringRef &Res) const {
  Elf_Dyn_iterator i = Elf_Dyn_iterator(dot_dynamic_sec->sh_entsize,
                                        reinterpret_cast<const char *>(Data.p));
  if (i == end_dynamic_table())
    report_fatal_error("getLibraryPath() called on iterator end");

  if (i->getTag() != ELF::DT_NEEDED)
    report_fatal_error("Invalid library_iterator");

  // This uses .dynstr to lookup the name of the DT_NEEDED entry.
  // THis works as long as DT_STRTAB == .dynstr. This is true most of
  // the time, but the specification allows exceptions.
  // TODO: This should really use DT_STRTAB instead. Doing this requires
  // reading the program headers.
  if (dot_dynstr_sec == NULL)
    report_fatal_error("Dynamic string table is missing");
  Res = getString(dot_dynstr_sec, i->getVal());
  return object_error::success;
}

template<class ELFT>
library_iterator ELFObjectFile<ELFT>::end_libraries_needed() const {
  Elf_Dyn_iterator e = end_dynamic_table();
  DataRefImpl DRI;
  DRI.p = reinterpret_cast<uintptr_t>(e.get());
  return library_iterator(LibraryRef(DRI, this));
}

template<class ELFT>
uint8_t ELFObjectFile<ELFT>::getBytesInAddress() const {
  return ELFT::Is64Bits ? 8 : 4;
}

template<class ELFT>
StringRef ELFObjectFile<ELFT>::getFileFormatName() const {
  switch(Header->e_ident[ELF::EI_CLASS]) {
  case ELF::ELFCLASS32:
    switch(Header->e_machine) {
    case ELF::EM_386:
      return "ELF32-i386";
    case ELF::EM_X86_64:
      return "ELF32-x86-64";
    case ELF::EM_ARM:
      return "ELF32-arm";
    case ELF::EM_HEXAGON:
      return "ELF32-hexagon";
    case ELF::EM_MIPS:
      return "ELF32-mips";
    default:
      return "ELF32-unknown";
    }
  case ELF::ELFCLASS64:
    switch(Header->e_machine) {
    case ELF::EM_386:
      return "ELF64-i386";
    case ELF::EM_X86_64:
      return "ELF64-x86-64";
    case ELF::EM_AARCH64:
      return "ELF64-aarch64";
    case ELF::EM_PPC64:
      return "ELF64-ppc64";
    case ELF::EM_S390:
      return "ELF64-s390";
    default:
      return "ELF64-unknown";
    }
  default:
    // FIXME: Proper error handling.
    report_fatal_error("Invalid ELFCLASS!");
  }
}

template<class ELFT>
unsigned ELFObjectFile<ELFT>::getArch() const {
  switch(Header->e_machine) {
  case ELF::EM_386:
    return Triple::x86;
  case ELF::EM_X86_64:
    return Triple::x86_64;
  case ELF::EM_AARCH64:
    return Triple::aarch64;
  case ELF::EM_ARM:
    return Triple::arm;
  case ELF::EM_HEXAGON:
    return Triple::hexagon;
  case ELF::EM_MIPS:
    return (ELFT::TargetEndianness == support::little) ?
           Triple::mipsel : Triple::mips;
  case ELF::EM_PPC64:
    return Triple::ppc64;
  case ELF::EM_S390:
    return Triple::systemz;
  default:
    return Triple::UnknownArch;
  }
}

template<class ELFT>
uint64_t ELFObjectFile<ELFT>::getNumSections() const {
  assert(Header && "Header not initialized!");
  if (Header->e_shnum == ELF::SHN_UNDEF) {
    assert(SectionHeaderTable && "SectionHeaderTable not initialized!");
    return SectionHeaderTable->sh_size;
  }
  return Header->e_shnum;
}

template<class ELFT>
uint64_t
ELFObjectFile<ELFT>::getStringTableIndex() const {
  if (Header->e_shnum == ELF::SHN_UNDEF) {
    if (Header->e_shstrndx == ELF::SHN_HIRESERVE)
      return SectionHeaderTable->sh_link;
    if (Header->e_shstrndx >= getNumSections())
      return 0;
  }
  return Header->e_shstrndx;
}

template<class ELFT>
template<typename T>
inline const T *
ELFObjectFile<ELFT>::getEntry(uint16_t Section, uint32_t Entry) const {
  return getEntry<T>(getSection(Section), Entry);
}

template<class ELFT>
template<typename T>
inline const T *
ELFObjectFile<ELFT>::getEntry(const Elf_Shdr * Section, uint32_t Entry) const {
  return reinterpret_cast<const T *>(
           base()
           + Section->sh_offset
           + (Entry * Section->sh_entsize));
}

template<class ELFT>
const typename ELFObjectFile<ELFT>::Elf_Sym *
ELFObjectFile<ELFT>::getSymbol(DataRefImpl Symb) const {
  return getEntry<Elf_Sym>(SymbolTableSections[Symb.d.b], Symb.d.a);
}

template<class ELFT>
const typename ELFObjectFile<ELFT>::Elf_Rel *
ELFObjectFile<ELFT>::getRel(DataRefImpl Rel) const {
  return getEntry<Elf_Rel>(Rel.w.b, Rel.w.c);
}

template<class ELFT>
const typename ELFObjectFile<ELFT>::Elf_Rela *
ELFObjectFile<ELFT>::getRela(DataRefImpl Rela) const {
  return getEntry<Elf_Rela>(Rela.w.b, Rela.w.c);
}

template<class ELFT>
const typename ELFObjectFile<ELFT>::Elf_Shdr *
ELFObjectFile<ELFT>::getSection(DataRefImpl Symb) const {
  const Elf_Shdr *sec = getSection(Symb.d.b);
  if (sec->sh_type != ELF::SHT_SYMTAB || sec->sh_type != ELF::SHT_DYNSYM)
    // FIXME: Proper error handling.
    report_fatal_error("Invalid symbol table section!");
  return sec;
}

template<class ELFT>
const typename ELFObjectFile<ELFT>::Elf_Shdr *
ELFObjectFile<ELFT>::getSection(uint32_t index) const {
  if (index == 0)
    return 0;
  if (!SectionHeaderTable || index >= getNumSections())
    // FIXME: Proper error handling.
    report_fatal_error("Invalid section index!");

  return reinterpret_cast<const Elf_Shdr *>(
         reinterpret_cast<const char *>(SectionHeaderTable)
         + (index * Header->e_shentsize));
}

template<class ELFT>
const char *ELFObjectFile<ELFT>::getString(uint32_t section,
                                           ELF::Elf32_Word offset) const {
  return getString(getSection(section), offset);
}

template<class ELFT>
const char *ELFObjectFile<ELFT>::getString(const Elf_Shdr *section,
                                           ELF::Elf32_Word offset) const {
  assert(section && section->sh_type == ELF::SHT_STRTAB && "Invalid section!");
  if (offset >= section->sh_size)
    // FIXME: Proper error handling.
    report_fatal_error("Symbol name offset outside of string table!");
  return (const char *)base() + section->sh_offset + offset;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getSymbolName(const Elf_Shdr *section,
                                              const Elf_Sym *symb,
                                              StringRef &Result) const {
  if (symb->st_name == 0) {
    const Elf_Shdr *section = getSection(symb);
    if (!section)
      Result = "";
    else
      Result = getString(dot_shstrtab_sec, section->sh_name);
    return object_error::success;
  }

  if (section == SymbolTableSections[0]) {
    // Symbol is in .dynsym, use .dynstr string table
    Result = getString(dot_dynstr_sec, symb->st_name);
  } else {
    // Use the default symbol table name section.
    Result = getString(dot_strtab_sec, symb->st_name);
  }
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getSectionName(const Elf_Shdr *section,
                                               StringRef &Result) const {
  Result = StringRef(getString(dot_shstrtab_sec, section->sh_name));
  return object_error::success;
}

template<class ELFT>
error_code ELFObjectFile<ELFT>::getSymbolVersion(const Elf_Shdr *section,
                                                 const Elf_Sym *symb,
                                                 StringRef &Version,
                                                 bool &IsDefault) const {
  // Handle non-dynamic symbols.
  if (section != SymbolTableSections[0]) {
    // Non-dynamic symbols can have versions in their names
    // A name of the form 'foo@V1' indicates version 'V1', non-default.
    // A name of the form 'foo@@V2' indicates version 'V2', default version.
    StringRef Name;
    error_code ec = getSymbolName(section, symb, Name);
    if (ec != object_error::success)
      return ec;
    size_t atpos = Name.find('@');
    if (atpos == StringRef::npos) {
      Version = "";
      IsDefault = false;
      return object_error::success;
    }
    ++atpos;
    if (atpos < Name.size() && Name[atpos] == '@') {
      IsDefault = true;
      ++atpos;
    } else {
      IsDefault = false;
    }
    Version = Name.substr(atpos);
    return object_error::success;
  }

  // This is a dynamic symbol. Look in the GNU symbol version table.
  if (dot_gnu_version_sec == NULL) {
    // No version table.
    Version = "";
    IsDefault = false;
    return object_error::success;
  }

  // Determine the position in the symbol table of this entry.
  const char *sec_start = (const char*)base() + section->sh_offset;
  size_t entry_index = ((const char*)symb - sec_start)/section->sh_entsize;

  // Get the corresponding version index entry
  const Elf_Versym *vs = getEntry<Elf_Versym>(dot_gnu_version_sec, entry_index);
  size_t version_index = vs->vs_index & ELF::VERSYM_VERSION;

  // Special markers for unversioned symbols.
  if (version_index == ELF::VER_NDX_LOCAL ||
      version_index == ELF::VER_NDX_GLOBAL) {
    Version = "";
    IsDefault = false;
    return object_error::success;
  }

  // Lookup this symbol in the version table
  LoadVersionMap();
  if (version_index >= VersionMap.size() || VersionMap[version_index].isNull())
    report_fatal_error("Symbol has version index without corresponding "
                       "define or reference entry");
  const VersionMapEntry &entry = VersionMap[version_index];

  // Get the version name string
  size_t name_offset;
  if (entry.isVerdef()) {
    // The first Verdaux entry holds the name.
    name_offset = entry.getVerdef()->getAux()->vda_name;
  } else {
    name_offset = entry.getVernaux()->vna_name;
  }
  Version = getString(dot_dynstr_sec, name_offset);

  // Set IsDefault
  if (entry.isVerdef()) {
    IsDefault = !(vs->vs_index & ELF::VERSYM_HIDDEN);
  } else {
    IsDefault = false;
  }

  return object_error::success;
}

/// This is a generic interface for retrieving GNU symbol version
/// information from an ELFObjectFile.
static inline error_code GetELFSymbolVersion(const ObjectFile *Obj,
                                             const SymbolRef &Sym,
                                             StringRef &Version,
                                             bool &IsDefault) {
  // Little-endian 32-bit
  if (const ELFObjectFile<ELFType<support::little, 4, false> > *ELFObj =
          dyn_cast<ELFObjectFile<ELFType<support::little, 4, false> > >(Obj))
    return ELFObj->getSymbolVersion(Sym, Version, IsDefault);

  // Big-endian 32-bit
  if (const ELFObjectFile<ELFType<support::big, 4, false> > *ELFObj =
          dyn_cast<ELFObjectFile<ELFType<support::big, 4, false> > >(Obj))
    return ELFObj->getSymbolVersion(Sym, Version, IsDefault);

  // Little-endian 64-bit
  if (const ELFObjectFile<ELFType<support::little, 8, true> > *ELFObj =
          dyn_cast<ELFObjectFile<ELFType<support::little, 8, true> > >(Obj))
    return ELFObj->getSymbolVersion(Sym, Version, IsDefault);

  // Big-endian 64-bit
  if (const ELFObjectFile<ELFType<support::big, 8, true> > *ELFObj =
          dyn_cast<ELFObjectFile<ELFType<support::big, 8, true> > >(Obj))
    return ELFObj->getSymbolVersion(Sym, Version, IsDefault);

  llvm_unreachable("Object passed to GetELFSymbolVersion() is not ELF");
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

}
}

#endif
