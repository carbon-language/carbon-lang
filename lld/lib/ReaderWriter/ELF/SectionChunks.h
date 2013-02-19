//===- lib/ReaderWriter/ELF/SectionChunks.h -------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_SECTION_CHUNKS_H
#define LLD_READER_WRITER_ELF_SECTION_CHUNKS_H

#include "Chunk.h"
#include "Layout.h"
#include "TargetHandler.h"
#include "Writer.h"

#include "lld/Core/DefinedAtom.h"
#include "lld/Core/range.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileOutputBuffer.h"

namespace lld {
namespace elf {
using namespace llvm::ELF;

/// \brief An ELF section.
template <class ELFT> class Section : public Chunk<ELFT> {
public:
  /// \param type the ELF SHT_* type of the section.
  Section(const ELFTargetInfo &ti, StringRef name,
          typename Chunk<ELFT>::Kind k = Chunk<ELFT>::K_ELFSection)
      : Chunk<ELFT>(name, k, ti),
        _flags(0),
        _entSize(0),
        _type(0),
        _link(0),
        _info(0),
        _segmentType(SHT_NULL) {}

  /// \brief Finalize the section contents before writing
  virtual void finalize() {}

  /// \brief Does this section have an output segment.
  virtual bool hasOutputSegment() {
    return false;
  }

  /// \brief Assign file offsets starting at offset.
  virtual void assignOffsets(uint64_t offset) {}

  /// \brief Assign virtual addresses starting at addr. Addr is modified to be
  /// the next available virtual address.
  virtual void assignVirtualAddress(uint64_t &addr) {}

  uint64_t getFlags() const { return _flags; }
  uint64_t getEntSize() const { return _entSize; }
  uint32_t getType() const { return _type; }
  uint32_t getLink() const { return _link; }
  uint32_t getInfo() const { return _info; }
  Layout::SegmentType getSegmentType() const { return _segmentType; }

  /// \brief convert the segment type to a String for diagnostics and printing
  /// purposes
  StringRef segmentKindToStr() const;

  // TODO: Move this down to AtomSection.
  virtual bool findAtomAddrByName(StringRef name, uint64_t &addr) {
    return false;
  }

  /// \brief Records the segmentType, that this section belongs to
  void setSegment(const Layout::SegmentType segmentType) {
    this->_segmentType = segmentType;
  }

  static bool classof(const Chunk<ELFT> *c) {
    return c->kind() == Chunk<ELFT>::K_ELFSection ||
           c->kind() == Chunk<ELFT>::K_AtomSection;
  }

protected:
  /// \brief ELF SHF_* flags.
  uint64_t _flags;
  /// \brief The size of each entity.
  uint64_t _entSize;
  /// \brief ELF SHT_* type.
  uint32_t _type;
  /// \brief sh_link field.
  uint32_t _link;
  /// \brief the sh_info field.
  uint32_t _info;
  /// \brief the output ELF segment type of this section.
  Layout::SegmentType _segmentType;
};

/// \brief A section containing atoms.
template <class ELFT> class AtomSection : public Section<ELFT> {
public:
  AtomSection(const ELFTargetInfo &ti, StringRef name, int32_t contentType,
              int32_t permissions, int32_t order)
      : Section<ELFT>(ti, name, Chunk<ELFT>::K_AtomSection),
        _contentType(contentType), _contentPermissions(permissions) {
    this->setOrder(order);
    switch (contentType) {
    case DefinedAtom::typeCode:
    case DefinedAtom::typeData:
    case DefinedAtom::typeConstant:
    case DefinedAtom::typeGOT:
    case DefinedAtom::typeStub:
    case DefinedAtom::typeResolver:
    case DefinedAtom::typeTLVInitialData:
      this->_type = SHT_PROGBITS;
      break;
    case DefinedAtom::typeZeroFill:
    case DefinedAtom::typeTLVInitialZeroFill:
      this->_type = SHT_NOBITS;
      break;
    }

    switch (permissions) {
    case DefinedAtom::permR__:
      this->_flags = SHF_ALLOC;
      break;
    case DefinedAtom::permR_X:
      this->_flags = SHF_ALLOC | SHF_EXECINSTR;
      break;
    case DefinedAtom::permRW_:
    case DefinedAtom::permRW_L:
      this->_flags = SHF_ALLOC | SHF_WRITE;
      if (_contentType == DefinedAtom::typeTLVInitialData ||
          _contentType == DefinedAtom::typeTLVInitialZeroFill)
        this->_flags |= SHF_TLS;
      break;
    case DefinedAtom::permRWX:
      this->_flags = SHF_ALLOC | SHF_WRITE | SHF_EXECINSTR;
      break;
    }
  }

  /// Align the offset to the required modulus defined by the atom alignment
  uint64_t alignOffset(uint64_t offset, DefinedAtom::Alignment &atomAlign);

  // \brief Append an atom to a Section. The atom gets pushed into a vector
  // contains the atom, the atom file offset, the atom virtual address
  // the atom file offset is aligned appropriately as set by the Reader
  const AtomLayout &appendAtom(const Atom *atom);

  /// \brief Set the virtual address of each Atom in the Section. This
  /// routine gets called after the linker fixes up the virtual address
  /// of the section
  virtual void assignVirtualAddress(uint64_t &addr) {
    for (auto &ai : _atoms) {
      ai->_virtualAddr = addr + ai->_fileOffset;
    }
  }

  /// \brief Set the file offset of each Atom in the section. This routine
  /// gets called after the linker fixes up the section offset
  virtual void assignOffsets(uint64_t offset) {
    for (auto &ai : _atoms) {
      ai->_fileOffset = offset + ai->_fileOffset;
    }
  }

  /// \brief Find the Atom address given a name, this is needed to to properly
  ///  apply relocation. The section class calls this to find the atom address
  ///  to fix the relocation
  virtual bool findAtomAddrByName(StringRef name, uint64_t &addr) {
    for (auto ai : _atoms) {
      if (ai->_atom->name() == name) {
        addr = ai->_virtualAddr;
        return true;
      }
    }
    return false;
  }

  /// \brief Does the Atom occupy any disk space
  bool occupiesNoDiskSpace() const {
    return _contentType == DefinedAtom::typeZeroFill;
  }

  /// \brief The permission of the section is the most permissive permission
  /// of all atoms that the section contains
  void setContentPermissions(int32_t perm) {
    _contentPermissions = std::max(perm, _contentPermissions);
  }

  /// \brief Return the raw flags, we need this to sort segments
  inline int64_t atomflags() const {
    return _contentPermissions;
  }

  /// Atom Iterators
  typedef typename std::vector<AtomLayout *>::iterator atom_iter;

  range<atom_iter> atoms() { return _atoms; }

  virtual void write(ELFWriter *writer, llvm::FileOutputBuffer &buffer);

  static bool classof(const Chunk<ELFT> *c) {
    return c->kind() == Chunk<ELFT>::K_AtomSection;
  }

protected:
  llvm::BumpPtrAllocator _alloc;
  int32_t _contentType;
  int32_t _contentPermissions;
  std::vector<AtomLayout *> _atoms;
};

/// Align the offset to the required modulus defined by the atom alignment
template <class ELFT>
uint64_t AtomSection<ELFT>::alignOffset(uint64_t offset,
                                        DefinedAtom::Alignment &atomAlign) {
  uint64_t requiredModulus = atomAlign.modulus;
  uint64_t align2 = 1u << atomAlign.powerOf2;
  uint64_t currentModulus = (offset % align2);
  uint64_t retOffset = offset;
  if (currentModulus != requiredModulus) {
    if (requiredModulus > currentModulus)
      retOffset += requiredModulus - currentModulus;
    else
      retOffset += align2 + requiredModulus - currentModulus;
  }
  return retOffset;
}

// \brief Append an atom to a Section. The atom gets pushed into a vector
// contains the atom, the atom file offset, the atom virtual address
// the atom file offset is aligned appropriately as set by the Reader
template <class ELFT>
const AtomLayout &AtomSection<ELFT>::appendAtom(const Atom *atom) {
  Atom::Definition atomType = atom->definition();
  const DefinedAtom *definedAtom = cast<DefinedAtom>(atom);

  DefinedAtom::Alignment atomAlign = definedAtom->alignment();
  uint64_t align2 = 1u << atomAlign.powerOf2;
  // Align the atom to the required modulus/ align the file offset and the
  // memory offset seperately this is required so that BSS symbols are handled
  // properly as the BSS symbols only occupy memory size and not file size
  uint64_t fOffset = alignOffset(this->fileSize(), atomAlign);
  uint64_t mOffset = alignOffset(this->memSize(), atomAlign);
  switch (atomType) {
  case Atom::definitionRegular:
    switch(definedAtom->contentType()) {
    case DefinedAtom::typeCode:
    case DefinedAtom::typeData:
    case DefinedAtom::typeConstant:
    case DefinedAtom::typeGOT:
    case DefinedAtom::typeStub:
    case DefinedAtom::typeResolver:
    case DefinedAtom::typeTLVInitialData:
      _atoms.push_back(new (_alloc) AtomLayout(atom, fOffset, 0));
      this->_fsize = fOffset + definedAtom->size();
      this->_msize = mOffset + definedAtom->size();
      DEBUG_WITH_TYPE("Section",
                      llvm::dbgs() << "[" << this->name() << " " << this << "] "
                                   << "Adding atom: " << atom->name() << "@"
                                   << fOffset << "\n");
      break;
    case DefinedAtom::typeZeroFill:
    case DefinedAtom::typeTLVInitialZeroFill:
      _atoms.push_back(new (_alloc) AtomLayout(atom, mOffset, 0));
      this->_msize = mOffset + definedAtom->size();
      break;
    default:
      llvm::dbgs() << definedAtom->contentType() << "\n";
      llvm_unreachable("Uexpected content type.");
    }
    break;
  default:
    llvm_unreachable("Expecting only definedAtoms being passed here");
    break;
  }
  // Set the section alignment to the largest alignment
  // std::max doesnot support uint64_t
  if (this->_align2 < align2)
    this->_align2 = align2;

  return *_atoms.back();
}

/// \brief convert the segment type to a String for diagnostics
///        and printing purposes
template <class ELFT> StringRef Section<ELFT>::segmentKindToStr() const {
  switch(_segmentType) {
  case llvm::ELF::PT_INTERP:
    return "INTERP";
  case llvm::ELF::PT_LOAD:
    return "LOAD";
  case llvm::ELF::PT_GNU_EH_FRAME:
    return "EH_FRAME";
  case llvm::ELF::PT_NOTE:
    return "NOTE";
  case llvm::ELF::PT_DYNAMIC:
    return "DYNAMIC";
  case llvm::ELF::PT_GNU_RELRO:
    return "RELRO";
  case llvm::ELF::PT_NULL:
    return "NULL";
  default:
    return "UNKNOWN";
  }
}

/// \brief Write the section and the atom contents to the buffer
template <class ELFT>
void AtomSection<ELFT>::write(ELFWriter *writer,
                              llvm::FileOutputBuffer &buffer) {
  uint8_t *chunkBuffer = buffer.getBufferStart();
  for (auto &ai : _atoms) {
    DEBUG_WITH_TYPE("Section",
                    llvm::dbgs() << "Writing atom: " << ai->_atom->name()
                                 << " | " << ai->_fileOffset << "\n");
    const DefinedAtom *definedAtom = cast<DefinedAtom>(ai->_atom);
    if (definedAtom->contentType() == DefinedAtom::typeZeroFill)
      continue;
    // Copy raw content of atom to file buffer.
    llvm::ArrayRef<uint8_t> content = definedAtom->rawContent();
    uint64_t contentSize = content.size();
    if (contentSize == 0)
      continue;
    uint8_t *atomContent = chunkBuffer + ai->_fileOffset;
    std::copy_n(content.data(), contentSize, atomContent);
    const TargetRelocationHandler<ELFT> &relHandler =
        this->_targetInfo.template getTargetHandler<ELFT>()
        .getRelocationHandler();
    for (const auto ref : *definedAtom)
      relHandler.applyRelocation(*writer, buffer, *ai, *ref);
  }
}

/// \brief A MergedSections represents a set of sections grouped by the same
/// name. The output file that gets written by the linker has sections grouped
/// by similiar names
template<class ELFT>
class MergedSections {
public:
  // Iterators
  typedef typename std::vector<Chunk<ELFT> *>::iterator ChunkIter;

  MergedSections(StringRef name);

  // Appends a section into the list of sections that are part of this Merged
  // Section
  void appendSection(Chunk<ELFT> *c);

  // Set the MergedSections is associated with a segment
  inline void setHasSegment() { _hasSegment = true; }

  /// Sets the ordinal
  inline void setOrdinal(uint64_t ordinal) {
    _ordinal = ordinal;
  }

  /// Sets the Memory size
  inline void setMemSize(uint64_t memsz) {
    _memSize = memsz;
  }

  /// Sets the size fo the merged Section
  inline void setSize(uint64_t fsiz) {
    _size = fsiz;
  }

  // The offset of the first section contained in the merged section is
  // contained here
  inline void setFileOffset(uint64_t foffset) {
    _fileOffset = foffset;
  }

  // Sets the starting address of the section
  inline void setAddr(uint64_t addr) {
    _virtualAddr = addr;
  }

  inline range<ChunkIter> sections() { return _sections; }

  // The below functions returns the properties of the MergeSection
  inline bool hasSegment() const { return _hasSegment; }

  inline StringRef name() const { return _name; }

  inline int64_t shinfo() const { return _shInfo; }

  inline uint64_t align2() const { return _align2; }

  inline int64_t link() const { return _link; }

  inline int64_t type() const { return _type; }

  inline uint64_t virtualAddr() const { return _virtualAddr; }

  inline int64_t ordinal() const { return _ordinal; }

  inline int64_t kind() const { return _kind; }

  inline uint64_t fileSize() const { return _size; }

  inline int64_t entsize() const { return _entSize; }

  inline uint64_t fileOffset() const { return _fileOffset; }

  inline int64_t flags() const { return _flags; }

  inline uint64_t memSize() { return _memSize; }

private:
  StringRef _name;
  bool _hasSegment;
  uint64_t _ordinal;
  uint64_t _flags;
  uint64_t _size;
  uint64_t _memSize;
  uint64_t _fileOffset;
  uint64_t _virtualAddr;
  int64_t _shInfo;
  int64_t _entSize;
  int64_t _link;
  uint64_t _align2;
  int64_t _kind;
  int64_t _type;
  std::vector<Chunk<ELFT> *> _sections;
};

/// MergedSections
template<class ELFT>
MergedSections<ELFT>::MergedSections(StringRef name)
  : _name(name)
  ,_hasSegment(false)
  ,_ordinal(0)
  ,_flags(0)
  ,_size(0)
  ,_memSize(0)
  ,_fileOffset(0)
  ,_virtualAddr(0)
  ,_shInfo(0)
  ,_entSize(0)
  ,_link(0)
  ,_align2(0)
  ,_kind(0)
  ,_type(0) { }
  

template<class ELFT>
void
MergedSections<ELFT>::appendSection(Chunk<ELFT> *c) {
  if (c->align2() > _align2)
    _align2 = c->align2();
  if (const auto section = dyn_cast<Section<ELFT>>(c)) {
    _link = section->getLink();
    _shInfo = section->getInfo();
    _entSize = section->getEntSize();
    _type = section->getType();
    if (_flags < section->getFlags())
      _flags = section->getFlags();
  }
  _kind = c->kind();
  _sections.push_back(c);
}

/// \brief The class represents the ELF String Table
template<class ELFT>
class StringTable : public Section<ELFT> {
public:
  StringTable(const ELFTargetInfo &, const char *str, int32_t order);

  uint64_t addString(StringRef symname);

  virtual void write(ELFWriter *writer, llvm::FileOutputBuffer &buffer);

private:
  std::vector<StringRef> _strings;

  struct StringRefMappingInfo {
    static StringRef getEmptyKey() { return StringRef(); }
    static StringRef getTombstoneKey() { return StringRef(" ", 0); }
    static unsigned getHashValue(StringRef const val) {
      return llvm::HashString(val);
    }
    static bool isEqual(StringRef const lhs, StringRef const rhs) {
      return lhs.equals(rhs);
    }
  };
  typedef typename llvm::DenseMap<StringRef, uint64_t,
                                  StringRefMappingInfo> StringMapT;
  typedef typename StringMapT::iterator StringMapTIter;
  StringMapT _stringMap;
};

template <class ELFT>
StringTable<ELFT>::StringTable(const ELFTargetInfo &ti, const char *str,
                               int32_t order)
    : Section<ELFT>(ti, str) {
  // the string table has a NULL entry for which
  // add an empty string
  _strings.push_back("");
  this->_fsize = 1;
  this->_align2 = 1;
  this->setOrder(order);
  this->_type = SHT_STRTAB;
}

template <class ELFT> uint64_t StringTable<ELFT>::addString(StringRef symname) {

  if (symname.size() == 0)
    return 0;
  StringMapTIter stringIter = _stringMap.find(symname);
  if (stringIter == _stringMap.end()) {
    _strings.push_back(symname);
    uint64_t offset = this->_fsize;
    this->_fsize += symname.size() + 1;
    _stringMap[symname] = offset;
    return offset;
  }
  return stringIter->second;
}

template <class ELFT>
void StringTable<ELFT>::write(ELFWriter *writer,
                              llvm::FileOutputBuffer &buffer) {
  uint8_t *chunkBuffer = buffer.getBufferStart();
  uint8_t *dest = chunkBuffer + this->fileOffset();
  for (auto si : _strings) {
    memcpy(dest, si.data(), si.size());
    dest += si.size();
    memcpy(dest, "", 1);
    dest += 1;
  }
}

/// \brief The SymbolTable class represents the symbol table in a ELF file
template<class ELFT>
class SymbolTable : public Section<ELFT> {
public:
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;

  SymbolTable(const ELFTargetInfo &ti, const char *str, int32_t order);

  void addSymbol(const Atom *atom, int32_t sectionIndex, uint64_t addr = 0);

  virtual void finalize();

  virtual void write(ELFWriter *writer, llvm::FileOutputBuffer &buffer);

  void setStringSection(StringTable<ELFT> *s) { _stringSection = s; }

private:
  StringTable<ELFT> *_stringSection;
  std::vector<Elf_Sym*> _symbolTable;
  llvm::BumpPtrAllocator _symbolAllocate;
};

/// ELF Symbol Table 
template <class ELFT>
SymbolTable<ELFT>::SymbolTable(const ELFTargetInfo &ti, const char *str,
                               int32_t order)
    : Section<ELFT>(ti, str) {
  this->setOrder(order);
  Elf_Sym *symbol = new (_symbolAllocate.Allocate<Elf_Sym>()) Elf_Sym;
  memset((void *)symbol, 0, sizeof(Elf_Sym));
  _symbolTable.push_back(symbol);
  this->_entSize = sizeof(Elf_Sym);
  this->_fsize = sizeof(Elf_Sym);
  this->_align2 = sizeof(void *);
  this->_type = SHT_SYMTAB;
}

template <class ELFT>
void SymbolTable<ELFT>::addSymbol(const Atom *atom, int32_t sectionIndex,
                                  uint64_t addr) {
  Elf_Sym *symbol = new(_symbolAllocate.Allocate<Elf_Sym>()) Elf_Sym;
  unsigned char binding = 0, type = 0;
  symbol->st_name = _stringSection->addString(atom->name());
  symbol->st_size = 0;
  symbol->st_shndx = sectionIndex;
  symbol->st_value = 0;
  symbol->st_other = llvm::ELF::STV_DEFAULT;
  if (const DefinedAtom *da = dyn_cast<const DefinedAtom>(atom)){
    symbol->st_size = da->size();
    DefinedAtom::ContentType ct;
    switch (ct = da->contentType()){
    case DefinedAtom::typeCode:
    case DefinedAtom::typeStub:
      symbol->st_value = addr;
      type = llvm::ELF::STT_FUNC;
      break;
    case DefinedAtom::typeResolver:
      symbol->st_value = addr;
      type = llvm::ELF::STT_GNU_IFUNC;
      break;
    case DefinedAtom::typeData:
    case DefinedAtom::typeConstant:
    case DefinedAtom::typeGOT:
      symbol->st_value = addr;
      type = llvm::ELF::STT_OBJECT;
      break;
    case DefinedAtom::typeZeroFill:
      type = llvm::ELF::STT_OBJECT;
      symbol->st_value = addr;
      break;
    case DefinedAtom::typeTLVInitialData:
    case DefinedAtom::typeTLVInitialZeroFill:
      type = llvm::ELF::STT_TLS;
      symbol->st_value = addr;
      break;
    default:
      type = llvm::ELF::STT_NOTYPE;
    }
    if (da->scope() == DefinedAtom::scopeTranslationUnit)
      binding = llvm::ELF::STB_LOCAL;
    else
      binding = llvm::ELF::STB_GLOBAL;
  } else if (const AbsoluteAtom *aa = dyn_cast<const AbsoluteAtom>(atom)){
    type = llvm::ELF::STT_OBJECT;
    symbol->st_shndx = llvm::ELF::SHN_ABS;
    switch (aa->scope()) {
    case AbsoluteAtom::scopeLinkageUnit:
      symbol->st_other = llvm::ELF::STV_HIDDEN;
      binding = llvm::ELF::STB_LOCAL;
      break;
    case AbsoluteAtom::scopeTranslationUnit:
      binding = llvm::ELF::STB_LOCAL;
      break;
    case AbsoluteAtom::scopeGlobal:
      binding = llvm::ELF::STB_GLOBAL;
      break;
    }
    symbol->st_value = addr;
  } else {
   symbol->st_value = 0;
   type = llvm::ELF::STT_NOTYPE;
   binding = llvm::ELF::STB_WEAK;
  }
  symbol->setBindingAndType(binding, type);
  _symbolTable.push_back(symbol);
  this->_fsize += sizeof(Elf_Sym);
}

template <class ELFT> void SymbolTable<ELFT>::finalize() {
  // sh_info should be one greater than last symbol with STB_LOCAL binding
  // we sort the symbol table to keep all local symbols at the beginning
  std::stable_sort(_symbolTable.begin(), _symbolTable.end(),
  [](const Elf_Sym *A, const Elf_Sym *B) {
     return A->getBinding() < B->getBinding();
  });
  uint16_t shInfo = 0;
  for (auto i : _symbolTable) {
    if (i->getBinding() != llvm::ELF::STB_LOCAL)
      break;
    shInfo++;
  }
  this->_info = shInfo;
  this->_link = _stringSection->ordinal();
}

template <class ELFT>
void SymbolTable<ELFT>::write(ELFWriter *writer,
                              llvm::FileOutputBuffer &buffer) {
  uint8_t *chunkBuffer = buffer.getBufferStart();
  uint8_t *dest = chunkBuffer + this->fileOffset();
  for (auto sti : _symbolTable) {
    memcpy(dest, sti, sizeof(Elf_Sym));
    dest += sizeof(Elf_Sym);
  }
}

template <class ELFT> class RelocationTable : public Section<ELFT> {
public:
  typedef llvm::object::Elf_Rel_Impl<ELFT, true> Elf_Rela;

  RelocationTable(const ELFTargetInfo &ti, StringRef str, int32_t order)
      : Section<ELFT>(ti, str) {
    this->setOrder(order);
    this->_entSize = sizeof(Elf_Rela);
    this->_align2 = llvm::alignOf<Elf_Rela>();
    this->_type = SHT_RELA;
    this->_flags = SHF_ALLOC;
  }

  void addRelocation(const DefinedAtom &da, const Reference &r) {
    _relocs.emplace_back(&da, &r);
    this->_fsize = _relocs.size() * sizeof(Elf_Rela);
    this->_msize = this->_fsize;
  }

  virtual void write(ELFWriter *writer, llvm::FileOutputBuffer &buffer) {
    uint8_t *chunkBuffer = buffer.getBufferStart();
    uint8_t *dest = chunkBuffer + this->fileOffset();
    for (const auto &rel : _relocs) {
      Elf_Rela *r = reinterpret_cast<Elf_Rela *>(dest);
      r->setSymbolAndType(0, rel.second->kind());
      r->r_offset =
          writer->addressOfAtom(rel.first) + rel.second->offsetInAtom();
      r->r_addend =
          writer->addressOfAtom(rel.second->target()) + rel.second->addend();
      dest += sizeof(Elf_Rela);
      DEBUG_WITH_TYPE("ELFRelocationTable", llvm::dbgs()
                      << "IRELATIVE relocation at " << rel.first->name() << "@"
                      << r->r_offset << " to " << rel.second->target()->name()
                      << "@" << r->r_addend << "\n");
    }
  }

private:
  std::vector<std::pair<const DefinedAtom *, const Reference *>> _relocs;
};
} // end namespace elf
} // end namespace lld

#endif
