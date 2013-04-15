//===- MachO.h - MachO object file implementation ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MachOObjectFile class, which binds the MachOObject
// class to the generic ObjectFile wrapper.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_MACHO_H
#define LLVM_OBJECT_MACHO_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Object/MachOFormat.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MachO.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace object {

using support::endianness;

template<endianness E, bool B>
struct MachOType {
  static const endianness TargetEndianness = E;
  static const bool Is64Bits = B;
};

template<endianness TargetEndianness>
struct MachOInt24Impl;

template<>
struct MachOInt24Impl<support::little> {
  uint8_t bytes[3];
  operator uint32_t() const {
    return (bytes[2] << 24) | (bytes[1] << 16) | bytes[0];
  }
};

template<>
struct MachOInt24Impl<support::big> {
  uint8_t bytes[3];
  operator uint32_t() const {
    return (bytes[0] << 24) | (bytes[1] << 16) | bytes[2];
  }
};

template<endianness TargetEndianness>
struct MachODataTypeTypedefHelperCommon {
  typedef support::detail::packed_endian_specific_integral
    <uint16_t, TargetEndianness, support::unaligned> MachOInt16;
  typedef support::detail::packed_endian_specific_integral
    <uint32_t, TargetEndianness, support::unaligned> MachOInt32;
  typedef support::detail::packed_endian_specific_integral
    <uint64_t, TargetEndianness, support::unaligned> MachOInt64;
  typedef MachOInt24Impl<TargetEndianness> MachOInt24;
};

#define LLVM_MACHOB_IMPORT_TYPES_TYPENAME(E)                                 \
typedef typename MachODataTypeTypedefHelperCommon<E>::MachOInt16 MachOInt16; \
typedef typename MachODataTypeTypedefHelperCommon<E>::MachOInt32 MachOInt32; \
typedef typename MachODataTypeTypedefHelperCommon<E>::MachOInt64 MachOInt64; \
typedef typename MachODataTypeTypedefHelperCommon<E>::MachOInt24 MachOInt24;

#define LLVM_MACHOB_IMPORT_TYPES(E)                                          \
typedef MachODataTypeTypedefHelperCommon<E>::MachOInt16 MachOInt16; \
typedef MachODataTypeTypedefHelperCommon<E>::MachOInt32 MachOInt32; \
typedef MachODataTypeTypedefHelperCommon<E>::MachOInt64 MachOInt64; \
typedef MachODataTypeTypedefHelperCommon<E>::MachOInt24 MachOInt24;

template<class MachOT>
struct MachODataTypeTypedefHelper;

template<endianness TargetEndianness>
struct MachODataTypeTypedefHelper<MachOType<TargetEndianness, false> > {
  typedef MachODataTypeTypedefHelperCommon<TargetEndianness> Base;
  typedef typename Base::MachOInt32 MachOIntPtr;
};

template<endianness TargetEndianness>
struct MachODataTypeTypedefHelper<MachOType<TargetEndianness, true> > {
  typedef MachODataTypeTypedefHelperCommon<TargetEndianness> Base;
  typedef typename Base::MachOInt64 MachOIntPtr;
};

#define LLVM_MACHO_IMPORT_TYPES(MachOT, E, B)                        \
LLVM_MACHOB_IMPORT_TYPES_TYPENAME(E)                                 \
typedef typename                                                     \
  MachODataTypeTypedefHelper <MachOT<E, B> >::MachOIntPtr MachOIntPtr;

namespace MachOFormat {
  struct SectionBase {
    char Name[16];
    char SegmentName[16];
  };

  template<class MachOT>
  struct Section;

  template<endianness TargetEndianness>
  struct Section<MachOType<TargetEndianness, false> > {
    LLVM_MACHOB_IMPORT_TYPES_TYPENAME(TargetEndianness)
    char Name[16];
    char SegmentName[16];
    MachOInt32 Address;
    MachOInt32 Size;
    MachOInt32 Offset;
    MachOInt32 Align;
    MachOInt32 RelocationTableOffset;
    MachOInt32 NumRelocationTableEntries;
    MachOInt32 Flags;
    MachOInt32 Reserved1;
    MachOInt32 Reserved2;
  };

  template<endianness TargetEndianness>
  struct Section<MachOType<TargetEndianness, true> > {
    LLVM_MACHOB_IMPORT_TYPES_TYPENAME(TargetEndianness)
    char Name[16];
    char SegmentName[16];
    MachOInt64 Address;
    MachOInt64 Size;
    MachOInt32 Offset;
    MachOInt32 Align;
    MachOInt32 RelocationTableOffset;
    MachOInt32 NumRelocationTableEntries;
    MachOInt32 Flags;
    MachOInt32 Reserved1;
    MachOInt32 Reserved2;
    MachOInt32 Reserved3;
  };

  struct MachOInt24 {
    uint8_t bytes[3];
    operator uint32_t() const {
      return (bytes[2] << 24) | (bytes[1] << 16) | bytes[0];
    }
  };

  template<bool HostIsLittleEndian, endianness TargetEndianness>
  struct RelocationEntry;

  template<>
  struct RelocationEntry<true, support::little> {
    LLVM_MACHOB_IMPORT_TYPES(support::little)
    MachOInt32 Address;
    MachOInt24 SymbolNum;
    unsigned PCRel : 1;
    unsigned Length : 2;
    unsigned External : 1;
    unsigned Type : 4;
  };

  template<>
  struct RelocationEntry<false, support::little> {
    LLVM_MACHOB_IMPORT_TYPES(support::little)
    MachOInt32 Address;
    MachOInt24 SymbolNum;
    unsigned Type : 4;
    unsigned External : 1;
    unsigned Length : 2;
    unsigned PCRel : 1;
  };

  template<>
  struct RelocationEntry<true, support::big> {
    LLVM_MACHOB_IMPORT_TYPES(support::big)
    MachOInt32 Address;
    MachOInt24 SymbolNum;
    unsigned Type : 4;
    unsigned External : 1;
    unsigned Length : 2;
    unsigned PCRel : 1;
  };

  template<>
  struct RelocationEntry<false, support::big> {
    LLVM_MACHOB_IMPORT_TYPES(support::big)
    MachOInt32 Address;
    MachOInt24 SymbolNum;
    unsigned PCRel : 1;
    unsigned Length : 2;
    unsigned External : 1;
    unsigned Type : 4;
  };

  template<bool HostIsLittleEndian, endianness TargetEndianness>
  struct ScatteredRelocationEntry;

  template<>
  struct ScatteredRelocationEntry<true, support::little> {
    LLVM_MACHOB_IMPORT_TYPES(support::little)
    MachOInt24 Address;
    unsigned Type : 4;
    unsigned Length : 2;
    unsigned PCRel : 1;
    unsigned Scattered : 1;
    MachOInt32 Value;
  };

  template<>
  struct ScatteredRelocationEntry<false, support::little> {
    LLVM_MACHOB_IMPORT_TYPES(support::little)
    MachOInt24 Address;
    unsigned Scattered : 1;
    unsigned PCRel : 1;
    unsigned Length : 2;
    unsigned Type : 4;
    MachOInt32 Value;
  };

  template<>
  struct ScatteredRelocationEntry<true, support::big> {
    LLVM_MACHOB_IMPORT_TYPES(support::big)
    unsigned Type : 4;
    unsigned Length : 2;
    unsigned PCRel : 1;
    unsigned Scattered : 1;
    MachOInt24 Address;
    MachOInt32 Value;
  };

  template<>
  struct ScatteredRelocationEntry<false, support::big> {
    LLVM_MACHOB_IMPORT_TYPES(support::big)
    unsigned Scattered : 1;
    unsigned PCRel : 1;
    unsigned Length : 2;
    unsigned Type : 4;
    MachOInt24 Address;
    MachOInt32 Value;
  };

  template<endianness TargetEndianness>
  struct SymbolTableEntryBase {
    LLVM_MACHOB_IMPORT_TYPES_TYPENAME(TargetEndianness)
    MachOInt32 StringIndex;
    uint8_t Type;
    uint8_t SectionIndex;
    MachOInt16 Flags;
  };

  template<class MachOT>
  struct SymbolTableEntry;

  template<endianness TargetEndianness, bool Is64Bits>
  struct SymbolTableEntry<MachOType<TargetEndianness, Is64Bits> > {
    LLVM_MACHO_IMPORT_TYPES(MachOType, TargetEndianness, Is64Bits)
    MachOInt32 StringIndex;
    uint8_t Type;
    uint8_t SectionIndex;
    MachOInt16 Flags;
    MachOIntPtr Value;
  };

  template<endianness TargetEndianness>
  struct LoadCommand {
    LLVM_MACHOB_IMPORT_TYPES_TYPENAME(TargetEndianness)
    MachOInt32 Type;
    MachOInt32 Size;
  };

  template<endianness TargetEndianness>
  struct SymtabLoadCommand {
    LLVM_MACHOB_IMPORT_TYPES_TYPENAME(TargetEndianness)
    MachOInt32 Type;
    MachOInt32 Size;
    MachOInt32 SymbolTableOffset;
    MachOInt32 NumSymbolTableEntries;
    MachOInt32 StringTableOffset;
    MachOInt32 StringTableSize;
  };

  template<class MachOT>
  struct SegmentLoadCommand;

  template<endianness TargetEndianness, bool Is64Bits>
  struct SegmentLoadCommand<MachOType<TargetEndianness, Is64Bits> > {
    LLVM_MACHO_IMPORT_TYPES(MachOType, TargetEndianness, Is64Bits)
    MachOInt32 Type;
    MachOInt32 Size;
    char Name[16];
    MachOIntPtr VMAddress;
    MachOIntPtr VMSize;
    MachOIntPtr FileOffset;
    MachOIntPtr FileSize;
    MachOInt32 MaxVMProtection;
    MachOInt32 InitialVMProtection;
    MachOInt32 NumSections;
    MachOInt32 Flags;
  };

  template<endianness TargetEndianness>
  struct LinkeditDataLoadCommand {
    LLVM_MACHOB_IMPORT_TYPES_TYPENAME(TargetEndianness)
    MachOInt32 Type;
    MachOInt32 Size;
    MachOInt32 DataOffset;
    MachOInt32 DataSize;
  };

  template<endianness TargetEndianness>
  struct Header {
    LLVM_MACHOB_IMPORT_TYPES_TYPENAME(TargetEndianness)
    MachOInt32 Magic;
    MachOInt32 CPUType;
    MachOInt32 CPUSubtype;
    MachOInt32 FileType;
    MachOInt32 NumLoadCommands;
    MachOInt32 SizeOfLoadCommands;
    MachOInt32 Flags;
  };
}

class MachOObjectFileBase : public ObjectFile {
public:
  typedef MachOFormat::SectionBase SectionBase;

  MachOObjectFileBase(MemoryBuffer *Object, bool IsLittleEndian, bool Is64Bits,
                      error_code &ec);

  virtual symbol_iterator begin_dynamic_symbols() const;
  virtual symbol_iterator end_dynamic_symbols() const;
  virtual library_iterator begin_libraries_needed() const;
  virtual library_iterator end_libraries_needed() const;

  virtual uint8_t getBytesInAddress() const;
  virtual StringRef getLoadName() const;

  bool is64Bit() const;
  void ReadULEB128s(uint64_t Index, SmallVectorImpl<uint64_t> &Out) const;
  unsigned getHeaderSize() const;
  StringRef getData(size_t Offset, size_t Size) const;

  static inline bool classof(const Binary *v) {
    return v->isMachO();
  }

protected:
  StringRef parseSegmentOrSectionName(const char *P) const;

  virtual error_code getSymbolValue(DataRefImpl Symb, uint64_t &Val) const;
  virtual error_code isSectionData(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionBSS(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionRequiredForExecution(DataRefImpl Sec,
                                                   bool &Res) const;
  virtual error_code isSectionVirtual(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionReadOnlyData(DataRefImpl Sec, bool &Res) const;
  virtual relocation_iterator getSectionRelBegin(DataRefImpl Sec) const;

  virtual error_code getRelocationNext(DataRefImpl Rel,
                                       RelocationRef &Res) const;

  virtual error_code getLibraryNext(DataRefImpl LibData, LibraryRef &Res) const;
  virtual error_code getLibraryPath(DataRefImpl LibData, StringRef &Res) const;
  virtual error_code getRelocationAdditionalInfo(DataRefImpl Rel,
                                                 int64_t &Res) const;

  std::size_t getSectionIndex(DataRefImpl Sec) const;

  typedef SmallVector<DataRefImpl, 1> SectionList;
  SectionList Sections;
};

template<endianness TargetEndianness>
class MachOObjectFileMiddle : public MachOObjectFileBase {
public:

  typedef MachOFormat::SymbolTableEntryBase<TargetEndianness>
    SymbolTableEntryBase;
  typedef MachOFormat::LinkeditDataLoadCommand<TargetEndianness>
    LinkeditDataLoadCommand;
  typedef MachOFormat::Header<TargetEndianness> Header;
  typedef MachOFormat::SymtabLoadCommand<TargetEndianness> SymtabLoadCommand;
  typedef MachOFormat::RelocationEntry<sys::IsLittleEndianHost, TargetEndianness> RelocationEntry;
  typedef MachOFormat::ScatteredRelocationEntry<sys::IsLittleEndianHost, TargetEndianness>
    ScatteredRelocationEntry;
  typedef MachOFormat::LoadCommand<TargetEndianness> LoadCommand;

  MachOObjectFileMiddle(MemoryBuffer *Object, bool Is64Bits, error_code &ec);

  const Header *getHeader() const;
  const LoadCommand *getLoadCommandInfo(unsigned Index) const;
  const RelocationEntry *getRelocation(DataRefImpl Rel) const;
  bool isRelocationScattered(const RelocationEntry *RE) const;
  bool isRelocationPCRel(const RelocationEntry *RE) const;
  unsigned getRelocationLength(const RelocationEntry *RE) const;
  unsigned getRelocationTypeImpl(const RelocationEntry *RE) const;

  void moveToNextSymbol(DataRefImpl &DRI) const;
  void printRelocationTargetName(const RelocationEntry *RE,
                                 raw_string_ostream &fmt) const;
  const SectionBase *getSectionBase(DataRefImpl DRI) const;
  const SymbolTableEntryBase *getSymbolTableEntryBase(DataRefImpl DRI) const;
  unsigned getCPUType() const;

  // In a MachO file, sections have a segment name. This is used in the .o
  // files. They have a single segment, but this field specifies which segment
  // a section should be put in in the final object.
  StringRef getSectionFinalSegmentName(DataRefImpl Sec) const;

  // Names are stored as 16 bytes. These returns the raw 16 bytes without
  // interpreting them as a C string.
  ArrayRef<char> getSectionRawName(DataRefImpl Sec) const;
  ArrayRef<char> getSectionRawFinalSegmentName(DataRefImpl Sec) const;

  virtual error_code getSymbolFlags(DataRefImpl Symb, uint32_t &Res) const;
  virtual error_code getSymbolNMTypeChar(DataRefImpl Symb, char &Res) const;
  virtual error_code getSymbolType(DataRefImpl Symb,
                                   SymbolRef::Type &Res) const;
  virtual error_code getSymbolName(DataRefImpl Symb, StringRef &Res) const;
  virtual error_code getSymbolSection(DataRefImpl Symb,
                                      section_iterator &Res) const;
  virtual error_code getSectionName(DataRefImpl Sec, StringRef &Res) const;
  virtual error_code getSymbolNext(DataRefImpl Symb, SymbolRef &Res) const;
  virtual symbol_iterator begin_symbols() const;
  virtual unsigned getArch() const;
  virtual StringRef getFileFormatName() const;
  virtual symbol_iterator end_symbols() const;
  virtual section_iterator end_sections() const;

  static bool classof(const Binary *v);

private:
  // Helper to advance a section or symbol iterator multiple increments at a
  // time.
  template<class T>
  static error_code advance(T &it, size_t Val);

  template<class T>
  static void advanceTo(T &it, size_t Val);
};

template<class MachOT>
struct MachOObjectFileHelperCommon;

template<endianness TargetEndianness, bool Is64Bits>
struct MachOObjectFileHelperCommon<MachOType<TargetEndianness, Is64Bits> > {
  typedef
    MachOFormat::SegmentLoadCommand<MachOType<TargetEndianness, Is64Bits> >
    SegmentLoadCommand;
  typedef MachOFormat::SymbolTableEntry<MachOType<TargetEndianness, Is64Bits> >
    SymbolTableEntry;
  typedef MachOFormat::Section<MachOType<TargetEndianness, Is64Bits> > Section;
};

template<class MachOT>
struct MachOObjectFileHelper;

template<endianness TargetEndianness>
struct MachOObjectFileHelper<MachOType<TargetEndianness, false> > :
    public MachOObjectFileHelperCommon<MachOType<TargetEndianness, false> > {
  static const macho::LoadCommandType SegmentLoadType = macho::LCT_Segment;
};

template<endianness TargetEndianness>
struct MachOObjectFileHelper<MachOType<TargetEndianness, true> > :
    public MachOObjectFileHelperCommon<MachOType<TargetEndianness, true> > {
  static const macho::LoadCommandType SegmentLoadType = macho::LCT_Segment64;
};

template<class MachOT>
class MachOObjectFile : public MachOObjectFileMiddle<MachOT::TargetEndianness> {
public:
  static const endianness TargetEndianness = MachOT::TargetEndianness;
  static const bool Is64Bits = MachOT::Is64Bits;

  typedef MachOObjectFileMiddle<MachOT::TargetEndianness> Base;
  typedef typename Base::RelocationEntry RelocationEntry;
  typedef typename Base::SectionBase SectionBase;
  typedef typename Base::SymbolTableEntryBase SymbolTableEntryBase;
  typedef typename Base::LoadCommand LoadCommand;

  typedef MachOObjectFileHelper<MachOT> Helper;
  static const macho::LoadCommandType SegmentLoadType = Helper::SegmentLoadType;
  typedef typename Helper::SegmentLoadCommand SegmentLoadCommand;
  typedef typename Helper::SymbolTableEntry SymbolTableEntry;
  typedef typename Helper::Section Section;

  MachOObjectFile(MemoryBuffer *Object, error_code &ec);
  static bool classof(const Binary *v);

  const Section *getSection(DataRefImpl DRI) const;
  const SymbolTableEntry *getSymbolTableEntry(DataRefImpl DRI) const;
  const RelocationEntry *getRelocation(DataRefImpl Rel) const;

  virtual error_code getSectionAddress(DataRefImpl Sec, uint64_t &Res) const;
  virtual error_code getSectionSize(DataRefImpl Sec, uint64_t &Res) const;
  virtual error_code getSectionContents(DataRefImpl Sec, StringRef &Res) const;
  virtual error_code getSectionAlignment(DataRefImpl Sec, uint64_t &Res) const;
  virtual error_code isSectionText(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionZeroInit(DataRefImpl Sec, bool &Res) const;
  virtual relocation_iterator getSectionRelEnd(DataRefImpl Sec) const;
  virtual error_code getRelocationAddress(DataRefImpl Rel, uint64_t &Res) const;
  virtual error_code getRelocationOffset(DataRefImpl Rel, uint64_t &Res) const;
  virtual error_code getRelocationSymbol(DataRefImpl Rel, SymbolRef &Res) const;
  virtual error_code getRelocationType(DataRefImpl Rel, uint64_t &Res) const;
  virtual error_code getRelocationTypeName(DataRefImpl Rel,
                                           SmallVectorImpl<char> &Result) const;
  virtual error_code getRelocationValueString(DataRefImpl Rel,
                                           SmallVectorImpl<char> &Result) const;
  virtual error_code getRelocationHidden(DataRefImpl Rel, bool &Result) const;
  virtual error_code getSymbolFileOffset(DataRefImpl Symb, uint64_t &Res) const;
  virtual error_code sectionContainsSymbol(DataRefImpl Sec, DataRefImpl Symb,
                                           bool &Result) const;
  virtual error_code getSymbolAddress(DataRefImpl Symb, uint64_t &Res) const;
  virtual error_code getSymbolSize(DataRefImpl Symb, uint64_t &Res) const;
  virtual error_code getSectionNext(DataRefImpl Sec, SectionRef &Res) const;
  virtual section_iterator begin_sections() const;
  void moveToNextSection(DataRefImpl &DRI) const;
};

typedef MachOObjectFileMiddle<support::little> MachOObjectFileLE;
typedef MachOObjectFileMiddle<support::big> MachOObjectFileBE;

typedef MachOObjectFile<MachOType<support::little, false> >
  MachOObjectFileLE32;
typedef MachOObjectFile<MachOType<support::big, false> >
  MachOObjectFileBE32;
typedef MachOObjectFile<MachOType<support::little, true> >
  MachOObjectFileLE64;
typedef MachOObjectFile<MachOType<support::big, true> >
  MachOObjectFileBE64;

template<endianness TargetEndianness>
MachOObjectFileMiddle<TargetEndianness>::MachOObjectFileMiddle(MemoryBuffer *O,
                                                               bool Is64Bits,
                                                               error_code &ec) :
  MachOObjectFileBase(O, TargetEndianness == support::little, Is64Bits, ec) {
}

template<endianness E>
const typename MachOObjectFileMiddle<E>::SymbolTableEntryBase *
MachOObjectFileMiddle<E>::getSymbolTableEntryBase(DataRefImpl DRI) const {
  const LoadCommand *L = getLoadCommandInfo(DRI.d.a);
  const SymtabLoadCommand *S = reinterpret_cast<const SymtabLoadCommand *>(L);

  unsigned Index = DRI.d.b;

  unsigned SymbolTableEntrySize = is64Bit() ?
    sizeof(MachOObjectFileLE64::SymbolTableEntry) :
    sizeof(MachOObjectFileLE32::SymbolTableEntry);

  uint64_t Offset = S->SymbolTableOffset + Index * SymbolTableEntrySize;
  StringRef Data = getData(Offset, SymbolTableEntrySize);
  return reinterpret_cast<const SymbolTableEntryBase*>(Data.data());
}

template<endianness E>
const typename MachOObjectFileMiddle<E>::Header *
MachOObjectFileMiddle<E>::getHeader() const {
  StringRef Data = getData(0, sizeof(Header));
  return reinterpret_cast<const Header*>(Data.data());
}

template<endianness E>
const typename MachOObjectFileMiddle<E>::LoadCommand *
MachOObjectFileMiddle<E>::getLoadCommandInfo(unsigned Index) const {
  assert(Index < getHeader()->NumLoadCommands);
  uint64_t Offset;
  uint64_t NewOffset = getHeaderSize();
  const LoadCommand *Load;
  unsigned I = 0;
  do {
    Offset = NewOffset;
    StringRef Data = getData(Offset, sizeof(MachOObjectFileLE::LoadCommand));
    Load = reinterpret_cast<const LoadCommand*>(Data.data());
    NewOffset = Offset + Load->Size;
    ++I;
  } while (I != Index + 1);

  return reinterpret_cast<const LoadCommand*>(Load);
}

template<endianness E>
const typename MachOObjectFileMiddle<E>::RelocationEntry *
MachOObjectFileMiddle<E>::getRelocation(DataRefImpl Rel) const {
  if (const MachOObjectFile<MachOType<E, true> > *O =
      dyn_cast<MachOObjectFile<MachOType<E, true> > >(this))
    return O->getRelocation(Rel);

  const MachOObjectFile<MachOType<E, false> > *O =
    cast<MachOObjectFile<MachOType<E, false> > >(this);
  return O->getRelocation(Rel);
}

template<endianness E>
bool
MachOObjectFileMiddle<E>::isRelocationScattered(const RelocationEntry *RE)
                                                                         const {
  if (this->getCPUType() == llvm::MachO::CPUTypeX86_64)
    return false;
  return RE->Address & macho::RF_Scattered;
}

template<endianness E>
bool
MachOObjectFileMiddle<E>::isRelocationPCRel(const RelocationEntry *RE) const {
  typedef MachOObjectFileMiddle<E> ObjType;
  if (isRelocationScattered(RE)) {
    const typename MachOObjectFileMiddle<E>::ScatteredRelocationEntry *SRE =
      reinterpret_cast<const typename ObjType::ScatteredRelocationEntry *>(RE);
    return SRE->PCRel;
  }
  return RE->PCRel;
}

template<endianness E>
unsigned
MachOObjectFileMiddle<E>::getRelocationLength(const RelocationEntry *RE) const {
  typedef MachOObjectFileMiddle<E> ObjType;
  if (isRelocationScattered(RE)) {
    const typename ObjType::ScatteredRelocationEntry *SRE =
      reinterpret_cast<const typename ObjType::ScatteredRelocationEntry *>(RE);
    return SRE->Length;
  }
  return RE->Length;
}

template<endianness E>
unsigned
MachOObjectFileMiddle<E>::getRelocationTypeImpl(const RelocationEntry *RE)
                                                                         const {
  typedef MachOObjectFileMiddle<E> ObjType;
  if (isRelocationScattered(RE)) {
    const typename ObjType::ScatteredRelocationEntry *SRE =
      reinterpret_cast<const typename ObjType::ScatteredRelocationEntry *>(RE);
    return SRE->Type;
  }
  return RE->Type;
}

// Helper to advance a section or symbol iterator multiple increments at a time.
template<endianness E>
template<class T>
error_code MachOObjectFileMiddle<E>::advance(T &it, size_t Val) {
  error_code ec;
  while (Val--) {
    it.increment(ec);
  }
  return ec;
}

template<endianness E>
template<class T>
void MachOObjectFileMiddle<E>::advanceTo(T &it, size_t Val) {
  if (error_code ec = advance(it, Val))
    report_fatal_error(ec.message());
}

template<endianness E>
void
MachOObjectFileMiddle<E>::printRelocationTargetName(const RelocationEntry *RE,
                                                raw_string_ostream &fmt) const {
  bool IsScattered = isRelocationScattered(RE);

  // Target of a scattered relocation is an address.  In the interest of
  // generating pretty output, scan through the symbol table looking for a
  // symbol that aligns with that address.  If we find one, print it.
  // Otherwise, we just print the hex address of the target.
  if (IsScattered) {
    uint32_t Val = RE->SymbolNum;

    error_code ec;
    for (symbol_iterator SI = begin_symbols(), SE = end_symbols(); SI != SE;
        SI.increment(ec)) {
      if (ec) report_fatal_error(ec.message());

      uint64_t Addr;
      StringRef Name;

      if ((ec = SI->getAddress(Addr)))
        report_fatal_error(ec.message());
      if (Addr != Val) continue;
      if ((ec = SI->getName(Name)))
        report_fatal_error(ec.message());
      fmt << Name;
      return;
    }

    // If we couldn't find a symbol that this relocation refers to, try
    // to find a section beginning instead.
    for (section_iterator SI = begin_sections(), SE = end_sections(); SI != SE;
         SI.increment(ec)) {
      if (ec) report_fatal_error(ec.message());

      uint64_t Addr;
      StringRef Name;

      if ((ec = SI->getAddress(Addr)))
        report_fatal_error(ec.message());
      if (Addr != Val) continue;
      if ((ec = SI->getName(Name)))
        report_fatal_error(ec.message());
      fmt << Name;
      return;
    }

    fmt << format("0x%x", Val);
    return;
  }

  StringRef S;
  bool isExtern = RE->External;
  uint64_t Val = RE->Address;

  if (isExtern) {
    symbol_iterator SI = begin_symbols();
    advanceTo(SI, Val);
    SI->getName(S);
  } else {
    section_iterator SI = begin_sections();
    advanceTo(SI, Val);
    SI->getName(S);
  }

  fmt << S;
}

template<endianness E>
const typename MachOObjectFileMiddle<E>::SectionBase *
MachOObjectFileMiddle<E>::getSectionBase(DataRefImpl DRI) const {
  uintptr_t CommandAddr =
    reinterpret_cast<uintptr_t>(getLoadCommandInfo(DRI.d.a));

  bool Is64 = is64Bit();
  unsigned SegmentLoadSize =
    Is64 ? sizeof(MachOObjectFileLE64::SegmentLoadCommand) :
           sizeof(MachOObjectFileLE32::SegmentLoadCommand);
  unsigned SectionSize = Is64 ? sizeof(MachOObjectFileLE64::Section) :
                                sizeof(MachOObjectFileLE32::Section);

  uintptr_t SectionAddr = CommandAddr + SegmentLoadSize + DRI.d.b * SectionSize;
  return reinterpret_cast<const SectionBase*>(SectionAddr);
}

template<endianness E>
unsigned MachOObjectFileMiddle<E>::getCPUType() const {
  return getHeader()->CPUType;
}

template<endianness E>
void MachOObjectFileMiddle<E>::moveToNextSymbol(DataRefImpl &DRI) const {
  uint32_t LoadCommandCount = getHeader()->NumLoadCommands;
  while (DRI.d.a < LoadCommandCount) {
    const LoadCommand *L = getLoadCommandInfo(DRI.d.a);
    if (L->Type == macho::LCT_Symtab) {
      const SymtabLoadCommand *S =
        reinterpret_cast<const SymtabLoadCommand *>(L);
      if (DRI.d.b < S->NumSymbolTableEntries)
        return;
    }

    DRI.d.a++;
    DRI.d.b = 0;
  }
}

template<endianness E>
StringRef
MachOObjectFileMiddle<E>::getSectionFinalSegmentName(DataRefImpl Sec) const {
  ArrayRef<char> Raw = getSectionRawFinalSegmentName(Sec);
  return parseSegmentOrSectionName(Raw.data());
}

template<endianness E>
ArrayRef<char>
MachOObjectFileMiddle<E>::getSectionRawName(DataRefImpl Sec) const {
  const SectionBase *Base = getSectionBase(Sec);
  return ArrayRef<char>(Base->Name);
}

template<endianness E>
ArrayRef<char>
MachOObjectFileMiddle<E>::getSectionRawFinalSegmentName(DataRefImpl Sec) const {
  const SectionBase *Base = getSectionBase(Sec);
  return ArrayRef<char>(Base->SegmentName);
}

template<endianness E>
error_code MachOObjectFileMiddle<E>::getSymbolFlags(DataRefImpl DRI,
                                                    uint32_t &Result) const {
  const SymbolTableEntryBase *Entry = getSymbolTableEntryBase(DRI);

  uint8_t MachOType = Entry->Type;
  uint16_t MachOFlags = Entry->Flags;

  // TODO: Correctly set SF_ThreadLocal
  Result = SymbolRef::SF_None;

  if ((MachOType & MachO::NlistMaskType) == MachO::NListTypeUndefined)
    Result |= SymbolRef::SF_Undefined;

  if (MachOFlags & macho::STF_StabsEntryMask)
    Result |= SymbolRef::SF_FormatSpecific;

  if (MachOType & MachO::NlistMaskExternal) {
    Result |= SymbolRef::SF_Global;
    if ((MachOType & MachO::NlistMaskType) == MachO::NListTypeUndefined)
      Result |= SymbolRef::SF_Common;
  }

  if (MachOFlags & (MachO::NListDescWeakRef | MachO::NListDescWeakDef))
    Result |= SymbolRef::SF_Weak;

  if ((MachOType & MachO::NlistMaskType) == MachO::NListTypeAbsolute)
    Result |= SymbolRef::SF_Absolute;

  return object_error::success;
}

template<endianness E>
error_code MachOObjectFileMiddle<E>::getSymbolType(DataRefImpl Symb,
                                                   SymbolRef::Type &Res) const {
  const SymbolTableEntryBase *Entry = getSymbolTableEntryBase(Symb);
  uint8_t n_type = Entry->Type;

  Res = SymbolRef::ST_Other;

  // If this is a STAB debugging symbol, we can do nothing more.
  if (n_type & MachO::NlistMaskStab) {
    Res = SymbolRef::ST_Debug;
    return object_error::success;
  }

  switch (n_type & MachO::NlistMaskType) {
    case MachO::NListTypeUndefined :
      Res = SymbolRef::ST_Unknown;
      break;
    case MachO::NListTypeSection :
      Res = SymbolRef::ST_Function;
      break;
  }
  return object_error::success;
}

template<endianness E>
error_code MachOObjectFileMiddle<E>::getSymbolName(DataRefImpl Symb,
                                                   StringRef &Res) const {
  const LoadCommand *L = getLoadCommandInfo(Symb.d.a);
  const SymtabLoadCommand *S = reinterpret_cast<const SymtabLoadCommand *>(L);
  StringRef StringTable = getData(S->StringTableOffset, S->StringTableSize);
  const SymbolTableEntryBase *Entry = getSymbolTableEntryBase(Symb);
  const char *Start = &StringTable.data()[Entry->StringIndex];
  Res = StringRef(Start);
  return object_error::success;
}

template<endianness E>
error_code
MachOObjectFileMiddle<E>::getSymbolSection(DataRefImpl Symb,
                                           section_iterator &Res) const {
  const SymbolTableEntryBase *Entry = getSymbolTableEntryBase(Symb);
  uint8_t index = Entry->SectionIndex;

  if (index == 0)
    Res = end_sections();
  else
    Res = section_iterator(SectionRef(Sections[index-1], this));

  return object_error::success;
}


template<endianness E>
error_code MachOObjectFileMiddle<E>::getSymbolNMTypeChar(DataRefImpl Symb,
                                                         char &Res) const {
  const SymbolTableEntryBase *Entry = getSymbolTableEntryBase(Symb);
  uint8_t Type = Entry->Type;
  uint16_t Flags = Entry->Flags;

  char Char;
  switch (Type & macho::STF_TypeMask) {
    case macho::STT_Undefined:
      Char = 'u';
      break;
    case macho::STT_Absolute:
    case macho::STT_Section:
      Char = 's';
      break;
    default:
      Char = '?';
      break;
  }

  if (Flags & (macho::STF_External | macho::STF_PrivateExtern))
    Char = toupper(static_cast<unsigned char>(Char));
  Res = Char;
  return object_error::success;
}

template<endianness E>
error_code
MachOObjectFileMiddle<E>::getSectionName(DataRefImpl Sec,
                                         StringRef &Result) const {
  ArrayRef<char> Raw = getSectionRawName(Sec);
  Result = parseSegmentOrSectionName(Raw.data());
  return object_error::success;
}

template<endianness E>
error_code MachOObjectFileMiddle<E>::getSymbolNext(DataRefImpl Symb,
                                                     SymbolRef &Res) const {
  Symb.d.b++;
  moveToNextSymbol(Symb);
  Res = SymbolRef(Symb, this);
  return object_error::success;
}

template<endianness E>
symbol_iterator MachOObjectFileMiddle<E>::begin_symbols() const {
  // DRI.d.a = segment number; DRI.d.b = symbol index.
  DataRefImpl DRI;
  moveToNextSymbol(DRI);
  return symbol_iterator(SymbolRef(DRI, this));
}

template<endianness E>
unsigned MachOObjectFileMiddle<E>::getArch() const {
  switch (getCPUType()) {
  case llvm::MachO::CPUTypeI386:
    return Triple::x86;
  case llvm::MachO::CPUTypeX86_64:
    return Triple::x86_64;
  case llvm::MachO::CPUTypeARM:
    return Triple::arm;
  case llvm::MachO::CPUTypePowerPC:
    return Triple::ppc;
  case llvm::MachO::CPUTypePowerPC64:
    return Triple::ppc64;
  default:
    return Triple::UnknownArch;
  }
}

template<endianness E>
StringRef MachOObjectFileMiddle<E>::getFileFormatName() const {
  unsigned CPUType = getCPUType();
  if (!is64Bit()) {
    switch (CPUType) {
    case llvm::MachO::CPUTypeI386:
      return "Mach-O 32-bit i386";
    case llvm::MachO::CPUTypeARM:
      return "Mach-O arm";
    case llvm::MachO::CPUTypePowerPC:
      return "Mach-O 32-bit ppc";
    default:
      assert((CPUType & llvm::MachO::CPUArchABI64) == 0 &&
             "64-bit object file when we're not 64-bit?");
      return "Mach-O 32-bit unknown";
    }
  }

  // Make sure the cpu type has the correct mask.
  assert((CPUType & llvm::MachO::CPUArchABI64)
	 == llvm::MachO::CPUArchABI64 &&
	 "32-bit object file when we're 64-bit?");

  switch (CPUType) {
  case llvm::MachO::CPUTypeX86_64:
    return "Mach-O 64-bit x86-64";
  case llvm::MachO::CPUTypePowerPC64:
    return "Mach-O 64-bit ppc64";
  default:
    return "Mach-O 64-bit unknown";
  }
}

template<endianness E>
symbol_iterator MachOObjectFileMiddle<E>::end_symbols() const {
  DataRefImpl DRI;
  DRI.d.a = getHeader()->NumLoadCommands;
  return symbol_iterator(SymbolRef(DRI, this));
}

template<endianness E>
section_iterator MachOObjectFileMiddle<E>::end_sections() const {
  DataRefImpl DRI;
  DRI.d.a = getHeader()->NumLoadCommands;
  return section_iterator(SectionRef(DRI, this));
}

template<endianness E>
bool MachOObjectFileMiddle<E>::classof(const Binary *v) {
  return isa<MachOObjectFile<MachOType<E, false> > >(v) ||
    isa<MachOObjectFile<MachOType<E, true> > >(v);
}

template<class MachOT>
MachOObjectFile<MachOT>::MachOObjectFile(MemoryBuffer *Object,
                                         error_code &ec) :
  MachOObjectFileMiddle<TargetEndianness>(Object, Is64Bits, ec) {
  DataRefImpl DRI;
  moveToNextSection(DRI);
  uint32_t LoadCommandCount = this->getHeader()->NumLoadCommands;
  while (DRI.d.a < LoadCommandCount) {
    this->Sections.push_back(DRI);
    DRI.d.b++;
    moveToNextSection(DRI);
  }
}

template<class MachOT>
bool MachOObjectFile<MachOT>::classof(const Binary *v) {
  return v->getType() ==
    Base::getMachOType(TargetEndianness == support::little, Is64Bits);
}

template<class MachOT>
const typename MachOObjectFile<MachOT>::Section *
MachOObjectFile<MachOT>::getSection(DataRefImpl DRI) const {
  const SectionBase *Addr = this->getSectionBase(DRI);
  return reinterpret_cast<const Section*>(Addr);
}

template<class MachOT>
const typename MachOObjectFile<MachOT>::SymbolTableEntry *
MachOObjectFile<MachOT>::getSymbolTableEntry(DataRefImpl DRI) const {
  const SymbolTableEntryBase *Base = this->getSymbolTableEntryBase(DRI);
  return reinterpret_cast<const SymbolTableEntry*>(Base);
}

template<class MachOT>
const typename MachOObjectFile<MachOT>::RelocationEntry *
MachOObjectFile<MachOT>::getRelocation(DataRefImpl Rel) const {
  const Section *Sect = getSection(this->Sections[Rel.d.b]);
  uint32_t RelOffset = Sect->RelocationTableOffset;
  uint64_t Offset = RelOffset + Rel.d.a * sizeof(RelocationEntry);
  StringRef Data = this->getData(Offset, sizeof(RelocationEntry));
  return reinterpret_cast<const RelocationEntry*>(Data.data());
}

template<class MachOT>
error_code
MachOObjectFile<MachOT>::getSectionAddress(DataRefImpl Sec,
                                           uint64_t &Res) const {
  const Section *Sect = getSection(Sec);
  Res = Sect->Address;
  return object_error::success;
}

template<class MachOT>
error_code
MachOObjectFile<MachOT>::getSectionSize(DataRefImpl Sec,
                                        uint64_t &Res) const {
  const Section *Sect = getSection(Sec);
  Res = Sect->Size;
  return object_error::success;
}

template<class MachOT>
error_code
MachOObjectFile<MachOT>::getSectionContents(DataRefImpl Sec,
                                            StringRef &Res) const {
  const Section *Sect = getSection(Sec);
  Res = this->getData(Sect->Offset, Sect->Size);
  return object_error::success;
}

template<class MachOT>
error_code
MachOObjectFile<MachOT>::getSectionAlignment(DataRefImpl Sec,
                                             uint64_t &Res) const {
  const Section *Sect = getSection(Sec);
  Res = uint64_t(1) << Sect->Align;
  return object_error::success;
}

template<class MachOT>
error_code
MachOObjectFile<MachOT>::isSectionText(DataRefImpl Sec, bool &Res) const {
  const Section *Sect = getSection(Sec);
  Res = Sect->Flags & macho::SF_PureInstructions;
  return object_error::success;
}

template<class MachOT>
error_code
MachOObjectFile<MachOT>::isSectionZeroInit(DataRefImpl Sec, bool &Res) const {
  const Section *Sect = getSection(Sec);
  unsigned SectionType = Sect->Flags & MachO::SectionFlagMaskSectionType;
  Res = SectionType == MachO::SectionTypeZeroFill ||
    SectionType == MachO::SectionTypeZeroFillLarge;
  return object_error::success;
}

template<class MachOT>
relocation_iterator
MachOObjectFile<MachOT>::getSectionRelEnd(DataRefImpl Sec) const {
  const Section *Sect = getSection(Sec);
  uint32_t LastReloc = Sect->NumRelocationTableEntries;
  DataRefImpl Ret;
  Ret.d.a = LastReloc;
  Ret.d.b = this->getSectionIndex(Sec);
  return relocation_iterator(RelocationRef(Ret, this));
}

template<class MachOT>
error_code
MachOObjectFile<MachOT>::getRelocationAddress(DataRefImpl Rel,
                                              uint64_t &Res) const {
  const Section *Sect = getSection(this->Sections[Rel.d.b]);
  uint64_t SectAddress = Sect->Address;
  const RelocationEntry *RE = getRelocation(Rel);

  uint64_t RelAddr;
  if (this->isRelocationScattered(RE))
    RelAddr = RE->Address & 0xFFFFFF;
  else
    RelAddr = RE->Address;

  Res = SectAddress + RelAddr;
  return object_error::success;
}

template<class MachOT>
error_code
MachOObjectFile<MachOT>::getRelocationOffset(DataRefImpl Rel,
                                             uint64_t &Res) const {
  const RelocationEntry *RE = getRelocation(Rel);
  if (this->isRelocationScattered(RE))
    Res = RE->Address & 0xFFFFFF;
  else
    Res = RE->Address;
  return object_error::success;
}

template<class MachOT>
error_code
MachOObjectFile<MachOT>::getRelocationSymbol(DataRefImpl Rel,
                                             SymbolRef &Res) const {
  const RelocationEntry *RE = getRelocation(Rel);
  uint32_t SymbolIdx = RE->SymbolNum;
  bool isExtern = RE->External;

  DataRefImpl Sym;
  this->moveToNextSymbol(Sym);
  if (isExtern) {
    for (unsigned i = 0; i < SymbolIdx; i++) {
      Sym.d.b++;
      this->moveToNextSymbol(Sym);
      assert(Sym.d.a < this->getHeader()->NumLoadCommands &&
             "Relocation symbol index out of range!");
    }
  }
  Res = SymbolRef(Sym, this);
  return object_error::success;
}

template<class MachOT>
error_code MachOObjectFile<MachOT>::getRelocationType(DataRefImpl Rel,
                                                      uint64_t &Res) const {
  const RelocationEntry *RE = getRelocation(Rel);
  Res = this->getRelocationTypeImpl(RE);
  return object_error::success;
}

template<class MachOT>
error_code
MachOObjectFile<MachOT>::getRelocationTypeName(DataRefImpl Rel,
                                          SmallVectorImpl<char> &Result) const {
    // TODO: Support scattered relocations.
  StringRef res;
  const RelocationEntry *RE = getRelocation(Rel);

  unsigned Arch = this->getArch();

  unsigned r_type = this->getRelocationTypeImpl(RE);

  switch (Arch) {
    case Triple::x86: {
      static const char *const Table[] =  {
        "GENERIC_RELOC_VANILLA",
        "GENERIC_RELOC_PAIR",
        "GENERIC_RELOC_SECTDIFF",
        "GENERIC_RELOC_PB_LA_PTR",
        "GENERIC_RELOC_LOCAL_SECTDIFF",
        "GENERIC_RELOC_TLV" };

      if (r_type > 6)
        res = "Unknown";
      else
        res = Table[r_type];
      break;
    }
    case Triple::x86_64: {
      static const char *const Table[] =  {
        "X86_64_RELOC_UNSIGNED",
        "X86_64_RELOC_SIGNED",
        "X86_64_RELOC_BRANCH",
        "X86_64_RELOC_GOT_LOAD",
        "X86_64_RELOC_GOT",
        "X86_64_RELOC_SUBTRACTOR",
        "X86_64_RELOC_SIGNED_1",
        "X86_64_RELOC_SIGNED_2",
        "X86_64_RELOC_SIGNED_4",
        "X86_64_RELOC_TLV" };

      if (r_type > 9)
        res = "Unknown";
      else
        res = Table[r_type];
      break;
    }
    case Triple::arm: {
      static const char *const Table[] =  {
        "ARM_RELOC_VANILLA",
        "ARM_RELOC_PAIR",
        "ARM_RELOC_SECTDIFF",
        "ARM_RELOC_LOCAL_SECTDIFF",
        "ARM_RELOC_PB_LA_PTR",
        "ARM_RELOC_BR24",
        "ARM_THUMB_RELOC_BR22",
        "ARM_THUMB_32BIT_BRANCH",
        "ARM_RELOC_HALF",
        "ARM_RELOC_HALF_SECTDIFF" };

      if (r_type > 9)
        res = "Unknown";
      else
        res = Table[r_type];
      break;
    }
    case Triple::ppc: {
      static const char *const Table[] =  {
        "PPC_RELOC_VANILLA",
        "PPC_RELOC_PAIR",
        "PPC_RELOC_BR14",
        "PPC_RELOC_BR24",
        "PPC_RELOC_HI16",
        "PPC_RELOC_LO16",
        "PPC_RELOC_HA16",
        "PPC_RELOC_LO14",
        "PPC_RELOC_SECTDIFF",
        "PPC_RELOC_PB_LA_PTR",
        "PPC_RELOC_HI16_SECTDIFF",
        "PPC_RELOC_LO16_SECTDIFF",
        "PPC_RELOC_HA16_SECTDIFF",
        "PPC_RELOC_JBSR",
        "PPC_RELOC_LO14_SECTDIFF",
        "PPC_RELOC_LOCAL_SECTDIFF" };

      res = Table[r_type];
      break;
    }
    case Triple::UnknownArch:
      res = "Unknown";
      break;
  }
  Result.append(res.begin(), res.end());
  return object_error::success;
}

template<class MachOT>
error_code
MachOObjectFile<MachOT>::getRelocationValueString(DataRefImpl Rel,
                                          SmallVectorImpl<char> &Result) const {
  const RelocationEntry *RE = getRelocation(Rel);

  unsigned Arch = this->getArch();
  bool IsScattered = this->isRelocationScattered(RE);

  std::string fmtbuf;
  raw_string_ostream fmt(fmtbuf);

  unsigned Type = this->getRelocationTypeImpl(RE);
  bool IsPCRel = this->isRelocationPCRel(RE);

  // Determine any addends that should be displayed with the relocation.
  // These require decoding the relocation type, which is triple-specific.

  // X86_64 has entirely custom relocation types.
  if (Arch == Triple::x86_64) {
    bool isPCRel = RE->PCRel;

    switch (Type) {
      case macho::RIT_X86_64_GOTLoad:   // X86_64_RELOC_GOT_LOAD
      case macho::RIT_X86_64_GOT: {     // X86_64_RELOC_GOT
        this->printRelocationTargetName(RE, fmt);
        fmt << "@GOT";
        if (isPCRel) fmt << "PCREL";
        break;
      }
      case macho::RIT_X86_64_Subtractor: { // X86_64_RELOC_SUBTRACTOR
        DataRefImpl RelNext = Rel;
        RelNext.d.a++;
        const RelocationEntry *RENext = getRelocation(RelNext);

        // X86_64_SUBTRACTOR must be followed by a relocation of type
        // X86_64_RELOC_UNSIGNED.
        // NOTE: Scattered relocations don't exist on x86_64.
        unsigned RType = RENext->Type;
        if (RType != 0)
          report_fatal_error("Expected X86_64_RELOC_UNSIGNED after "
                             "X86_64_RELOC_SUBTRACTOR.");

        // The X86_64_RELOC_UNSIGNED contains the minuend symbol,
        // X86_64_SUBTRACTOR contains to the subtrahend.
        this->printRelocationTargetName(RENext, fmt);
        fmt << "-";
        this->printRelocationTargetName(RE, fmt);
        break;
      }
      case macho::RIT_X86_64_TLV:
        this->printRelocationTargetName(RE, fmt);
        fmt << "@TLV";
        if (isPCRel) fmt << "P";
        break;
      case macho::RIT_X86_64_Signed1: // X86_64_RELOC_SIGNED1
        this->printRelocationTargetName(RE, fmt);
        fmt << "-1";
        break;
      case macho::RIT_X86_64_Signed2: // X86_64_RELOC_SIGNED2
        this->printRelocationTargetName(RE, fmt);
        fmt << "-2";
        break;
      case macho::RIT_X86_64_Signed4: // X86_64_RELOC_SIGNED4
        this->printRelocationTargetName(RE, fmt);
        fmt << "-4";
        break;
      default:
        this->printRelocationTargetName(RE, fmt);
        break;
    }
  // X86 and ARM share some relocation types in common.
  } else if (Arch == Triple::x86 || Arch == Triple::arm) {
    // Generic relocation types...
    switch (Type) {
      case macho::RIT_Pair: // GENERIC_RELOC_PAIR - prints no info
        return object_error::success;
      case macho::RIT_Difference: { // GENERIC_RELOC_SECTDIFF
        DataRefImpl RelNext = Rel;
        RelNext.d.a++;
        const RelocationEntry *RENext = getRelocation(RelNext);

        // X86 sect diff's must be followed by a relocation of type
        // GENERIC_RELOC_PAIR.
        bool isNextScattered = (Arch != Triple::x86_64) &&
                               (RENext->Address & macho::RF_Scattered);
        unsigned RType;
        if (isNextScattered)
          RType = (RENext->Address >> 24) & 0xF;
        else
          RType = RENext->Type;
        if (RType != 1)
          report_fatal_error("Expected GENERIC_RELOC_PAIR after "
                             "GENERIC_RELOC_SECTDIFF.");

        this->printRelocationTargetName(RE, fmt);
        fmt << "-";
        this->printRelocationTargetName(RENext, fmt);
        break;
      }
    }

    if (Arch == Triple::x86) {
      // All X86 relocations that need special printing were already
      // handled in the generic code.
      switch (Type) {
        case macho::RIT_Generic_LocalDifference:{// GENERIC_RELOC_LOCAL_SECTDIFF
          DataRefImpl RelNext = Rel;
          RelNext.d.a++;
          const RelocationEntry *RENext = getRelocation(RelNext);

          // X86 sect diff's must be followed by a relocation of type
          // GENERIC_RELOC_PAIR.
          bool isNextScattered = (Arch != Triple::x86_64) &&
                               (RENext->Address & macho::RF_Scattered);
          unsigned RType;
          if (isNextScattered)
            RType = (RENext->Address >> 24) & 0xF;
          else
            RType = RENext->Type;
          if (RType != 1)
            report_fatal_error("Expected GENERIC_RELOC_PAIR after "
                               "GENERIC_RELOC_LOCAL_SECTDIFF.");

          this->printRelocationTargetName(RE, fmt);
          fmt << "-";
          this->printRelocationTargetName(RENext, fmt);
          break;
        }
        case macho::RIT_Generic_TLV: {
          this->printRelocationTargetName(RE, fmt);
          fmt << "@TLV";
          if (IsPCRel) fmt << "P";
          break;
        }
        default:
          this->printRelocationTargetName(RE, fmt);
      }
    } else { // ARM-specific relocations
      switch (Type) {
        case macho::RIT_ARM_Half:             // ARM_RELOC_HALF
        case macho::RIT_ARM_HalfDifference: { // ARM_RELOC_HALF_SECTDIFF
          // Half relocations steal a bit from the length field to encode
          // whether this is an upper16 or a lower16 relocation.
          bool isUpper;
          if (IsScattered)
            isUpper = (RE->Address >> 28) & 1;
          else
            isUpper = (RE->Length >> 1) & 1;

          if (isUpper)
            fmt << ":upper16:(";
          else
            fmt << ":lower16:(";
          this->printRelocationTargetName(RE, fmt);

          DataRefImpl RelNext = Rel;
          RelNext.d.a++;
          const RelocationEntry *RENext = getRelocation(RelNext);

          // ARM half relocs must be followed by a relocation of type
          // ARM_RELOC_PAIR.
          bool isNextScattered = (Arch != Triple::x86_64) &&
                                 (RENext->Address & macho::RF_Scattered);
          unsigned RType;
          if (isNextScattered)
            RType = (RENext->Address >> 24) & 0xF;
          else
            RType = RENext->Type;

          if (RType != 1)
            report_fatal_error("Expected ARM_RELOC_PAIR after "
                               "GENERIC_RELOC_HALF");

          // NOTE: The half of the target virtual address is stashed in the
          // address field of the secondary relocation, but we can't reverse
          // engineer the constant offset from it without decoding the movw/movt
          // instruction to find the other half in its immediate field.

          // ARM_RELOC_HALF_SECTDIFF encodes the second section in the
          // symbol/section pointer of the follow-on relocation.
          if (Type == macho::RIT_ARM_HalfDifference) {
            fmt << "-";
            this->printRelocationTargetName(RENext, fmt);
          }

          fmt << ")";
          break;
        }
        default: {
          this->printRelocationTargetName(RE, fmt);
        }
      }
    }
  } else
    this->printRelocationTargetName(RE, fmt);

  fmt.flush();
  Result.append(fmtbuf.begin(), fmtbuf.end());
  return object_error::success;
}

template<class MachOT>
error_code
MachOObjectFile<MachOT>::getRelocationHidden(DataRefImpl Rel,
                                             bool &Result) const {
  const RelocationEntry *RE = getRelocation(Rel);
  unsigned Arch = this->getArch();
  unsigned Type = this->getRelocationTypeImpl(RE);

  Result = false;

  // On arches that use the generic relocations, GENERIC_RELOC_PAIR
  // is always hidden.
  if (Arch == Triple::x86 || Arch == Triple::arm) {
    if (Type == macho::RIT_Pair) Result = true;
  } else if (Arch == Triple::x86_64) {
    // On x86_64, X86_64_RELOC_UNSIGNED is hidden only when it follows
    // an X864_64_RELOC_SUBTRACTOR.
    if (Type == macho::RIT_X86_64_Unsigned && Rel.d.a > 0) {
      DataRefImpl RelPrev = Rel;
      RelPrev.d.a--;
      const RelocationEntry *REPrev = this->getRelocation(RelPrev);

      unsigned PrevType = REPrev->Type;

      if (PrevType == macho::RIT_X86_64_Subtractor) Result = true;
    }
  }

  return object_error::success;
}

template<class MachOT>
error_code
MachOObjectFile<MachOT>::getSymbolFileOffset(DataRefImpl Symb,
                                             uint64_t &Res) const {
  const SymbolTableEntry *Entry = getSymbolTableEntry(Symb);
  Res = Entry->Value;
  if (Entry->SectionIndex) {
    const Section *Sec =
      this->getSection(this->Sections[Entry->SectionIndex-1]);
    Res += Sec->Offset - Sec->Address;
  }

  return object_error::success;
}

template<class MachOT>
error_code
MachOObjectFile<MachOT>::sectionContainsSymbol(DataRefImpl Sec,
                                               DataRefImpl Symb,
                                               bool &Result) const {
  SymbolRef::Type ST;
  this->getSymbolType(Symb, ST);
  if (ST == SymbolRef::ST_Unknown) {
    Result = false;
    return object_error::success;
  }

  uint64_t SectBegin, SectEnd;
  getSectionAddress(Sec, SectBegin);
  getSectionSize(Sec, SectEnd);
  SectEnd += SectBegin;

  const SymbolTableEntry *Entry = getSymbolTableEntry(Symb);
  uint64_t SymAddr= Entry->Value;
  Result = (SymAddr >= SectBegin) && (SymAddr < SectEnd);

  return object_error::success;
}

template<class MachOT>
error_code MachOObjectFile<MachOT>::getSymbolAddress(DataRefImpl Symb,
                                                     uint64_t &Res) const {
  const SymbolTableEntry *Entry = getSymbolTableEntry(Symb);
  Res = Entry->Value;
  return object_error::success;
}

template<class MachOT>
error_code MachOObjectFile<MachOT>::getSymbolSize(DataRefImpl DRI,
                                                    uint64_t &Result) const {
  uint32_t LoadCommandCount = this->getHeader()->NumLoadCommands;
  uint64_t BeginOffset;
  uint64_t EndOffset = 0;
  uint8_t SectionIndex;

  const SymbolTableEntry *Entry = getSymbolTableEntry(DRI);
  BeginOffset = Entry->Value;
  SectionIndex = Entry->SectionIndex;
  if (!SectionIndex) {
    uint32_t flags = SymbolRef::SF_None;
    this->getSymbolFlags(DRI, flags);
    if (flags & SymbolRef::SF_Common)
      Result = Entry->Value;
    else
      Result = UnknownAddressOrSize;
    return object_error::success;
  }
  // Unfortunately symbols are unsorted so we need to touch all
  // symbols from load command
  DRI.d.b = 0;
  uint32_t Command = DRI.d.a;
  while (Command == DRI.d.a) {
    this->moveToNextSymbol(DRI);
    if (DRI.d.a < LoadCommandCount) {
      Entry = getSymbolTableEntry(DRI);
      if (Entry->SectionIndex == SectionIndex && Entry->Value > BeginOffset)
        if (!EndOffset || Entry->Value < EndOffset)
          EndOffset = Entry->Value;
    }
    DRI.d.b++;
  }
  if (!EndOffset) {
    uint64_t Size;
    this->getSectionSize(this->Sections[SectionIndex-1], Size);
    this->getSectionAddress(this->Sections[SectionIndex-1], EndOffset);
    EndOffset += Size;
  }
  Result = EndOffset - BeginOffset;
  return object_error::success;
}

template<class MachOT>
error_code MachOObjectFile<MachOT>::getSectionNext(DataRefImpl Sec,
                                                   SectionRef &Res) const {
  Sec.d.b++;
  moveToNextSection(Sec);
  Res = SectionRef(Sec, this);
  return object_error::success;
}

template<class MachOT>
section_iterator MachOObjectFile<MachOT>::begin_sections() const {
  DataRefImpl DRI;
  moveToNextSection(DRI);
  return section_iterator(SectionRef(DRI, this));
}

template<class MachOT>
void MachOObjectFile<MachOT>::moveToNextSection(DataRefImpl &DRI) const {
  uint32_t LoadCommandCount = this->getHeader()->NumLoadCommands;
  while (DRI.d.a < LoadCommandCount) {
    const LoadCommand *Command = this->getLoadCommandInfo(DRI.d.a);
    if (Command->Type == SegmentLoadType) {
      const SegmentLoadCommand *SegmentLoadCmd =
        reinterpret_cast<const SegmentLoadCommand*>(Command);
      if (DRI.d.b < SegmentLoadCmd->NumSections)
        return;
    }

    DRI.d.a++;
    DRI.d.b = 0;
  }
}

}
}

#endif

