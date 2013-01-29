//===- lib/ReaderWriter/ELF/ELFSectionChunks.h -----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_SECTION_CHUNKS_H_
#define LLD_READER_WRITER_ELF_SECTION_CHUNKS_H_

#include "lld/Core/DefinedAtom.h"
#include "lld/Core/range.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringRef.h"

#include "llvm/Object/ELF.h"

#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileOutputBuffer.h"

#include "ELFChunk.h"
#include "ELFLayout.h"
#include "ELFWriter.h"

namespace lld {
namespace elf {

/// \brief A section contains a set of atoms that have similiar properties
///        The atoms that have similiar properties are merged to form a section
template<class ELFT>
class Section : public Chunk<ELFT> {
public:
  // The Kind of section that the object represents
  enum SectionKind {
    K_Default,
    K_Target, // The section is handed over to the target
    K_SymbolTable,
    K_StringTable,
  };
  // Create a section object, the section is set to the default type if the
  // caller doesnot set it
  Section(const llvm::StringRef sectionName,
          const int32_t contentType,
          const int32_t contentPermissions,
          const int32_t order,
          const SectionKind kind = K_Default);

  /// return the section kind
  inline SectionKind sectionKind() const {
    return _sectionKind;
  }

  /// Align the offset to the required modulus defined by the atom alignment
  uint64_t alignOffset(uint64_t offset, DefinedAtom::Alignment &atomAlign);

  // \brief Append an atom to a Section. The atom gets pushed into a vector
  // contains the atom, the atom file offset, the atom virtual address
  // the atom file offset is aligned appropriately as set by the Reader
  void appendAtom(const Atom *atom);

  /// \brief Set the virtual address of each Atom in the Section. This
  /// routine gets called after the linker fixes up the virtual address
  /// of the section
  inline void assignVirtualAddress(uint64_t &addr) {
    for (auto &ai : _atoms) {
      ai._virtualAddr = addr + ai._fileOffset;
    }
    addr += this->memSize();
  }

  /// \brief Set the file offset of each Atom in the section. This routine
  /// gets called after the linker fixes up the section offset
  inline void assignOffsets(uint64_t offset) {
    for (auto &ai : _atoms) {
      ai._fileOffset = offset + ai._fileOffset;
    }
  }

  /// \brief Find the Atom address given a name, this is needed to to properly
  ///  apply relocation. The section class calls this to find the atom address
  ///  to fix the relocation
  inline bool findAtomAddrByName(const llvm::StringRef name, uint64_t &addr) {
    for (auto ai : _atoms) {
      if (ai._atom->name() == name) {
        addr = ai._virtualAddr;
        return true;
      }
    }
    return false;
  }

  /// \brief Does the Atom occupy any disk space
  inline bool occupiesNoDiskSpace() const {
    return _contentType == DefinedAtom::typeZeroFill;
  }

  /// \brief The permission of the section is the most permissive permission
  /// of all atoms that the section contains
  inline void setContentPermissions(int32_t perm) {
    _contentPermissions = std::max(perm, _contentPermissions);
  }

  /// \brief Get the section flags, defined by the permissions of the section
  int64_t flags();

  /// \brief Return the section type, the returned value is recorded in the
  /// sh_type field of the Section Header
  int type();

  /// \brief convert the segment type to a String for diagnostics
  ///        and printing purposes
  llvm::StringRef segmentKindToStr() const;

  /// \brief Return the raw flags, we need this to sort segments
  inline int64_t atomflags() const {
    return _contentPermissions;
  }

  /// \brief Returns the section link field, the returned value is
  ///        recorded in the sh_link field of the Section Header
  inline int link() const {
    return _link;
  }

  inline void setLink(int32_t link) {
    _link = link;
  }

  /// \brief Returns the section entsize field, the returned value is
  ///        recorded in the sh_entsize field of the Section Header
  inline int entsize() const {
    return _entSize;
  }

  /// \brief Returns the shinfo field, the returned value is
  ///        recorded in the sh_info field of the Section Header
  inline int shinfo() const {
    return _shInfo;
  }

  /// \brief Records the segmentType, that this section belongs to
  inline void setSegment(const ELFLayout::SegmentType segmentType) {
    _segmentType = segmentType;
  }

  /// \brief for LLVM style RTTI information
  static inline bool classof(const Chunk<ELFT> *c) {
    return c->kind() == Chunk<ELFT>::K_ELFSection;
  }

  /// \brief Finalize the section contents before writing
  inline void finalize() { }

  /// \brief Write the section and the atom contents to the buffer
  void write(ELFWriter *writer, llvm::FileOutputBuffer &buffer);

  /// Atom Iterators
  typedef typename std::vector<AtomLayout>::iterator atom_iter;

  range<atom_iter> atoms() { return _atoms; }

protected:
  int32_t _contentType;
  int32_t _contentPermissions;
  SectionKind _sectionKind;
  std::vector<AtomLayout> _atoms;
  ELFLayout::SegmentType _segmentType;
  int64_t _entSize;
  int64_t _shInfo;
  int64_t _link;
};

// Create a section object, the section is set to the default type if the
// caller doesnot set it
template<class ELFT>
Section<ELFT>::Section(const StringRef sectionName,
                       const int32_t contentType,
                       const int32_t contentPermissions,
                       const int32_t order,
                       const SectionKind kind)
  : Chunk<ELFT>(sectionName, Chunk<ELFT>::K_ELFSection)
  , _contentType(contentType)
  , _contentPermissions(contentPermissions)
  , _sectionKind(kind)
  , _entSize(0)
  , _shInfo(0)
  , _link(0) {
  this->setOrder(order);
}

/// Align the offset to the required modulus defined by the atom alignment
template<class ELFT>
uint64_t 
Section<ELFT>::alignOffset(uint64_t offset, DefinedAtom::Alignment &atomAlign) {
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
template<class ELFT>
void 
Section<ELFT>::appendAtom(const Atom *atom) {
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
    case  DefinedAtom::typeCode:
    case  DefinedAtom::typeData:
    case  DefinedAtom::typeConstant:
      _atoms.push_back(AtomLayout(atom, fOffset, 0));
      this->_fsize = fOffset + definedAtom->size();
      this->_msize = mOffset + definedAtom->size();
      break;
    case  DefinedAtom::typeZeroFill:
      _atoms.push_back(AtomLayout(atom, mOffset, 0));
      this->_msize = mOffset + definedAtom->size();
      break;
    default:
      this->_fsize = fOffset + definedAtom->size();
      this->_msize = mOffset + definedAtom->size();
      break;
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
}

/// \brief Get the section flags, defined by the permissions of the section
template<class ELFT>
int64_t 
Section<ELFT>::flags() {
  switch (_contentPermissions) {
  case DefinedAtom::perm___:
    return 0;

  case DefinedAtom::permR__:
      return llvm::ELF::SHF_ALLOC;

  case DefinedAtom::permR_X:
      return llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_EXECINSTR;

  case DefinedAtom::permRW_:
  case DefinedAtom::permRW_L:
      return llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_WRITE;

  case DefinedAtom::permRWX:
      return llvm::ELF::SHF_ALLOC |
              llvm::ELF::SHF_WRITE |
              llvm::ELF::SHF_EXECINSTR;

  default:
      break;
  }
  return llvm::ELF::SHF_ALLOC;
}

/// \brief Return the section type, the returned value is recorded in the
/// sh_type field of the Section Header

template<class ELFT>
int 
Section<ELFT>::type() {
  switch (_contentType) {
  case DefinedAtom::typeCode:
  case DefinedAtom::typeData:
  case DefinedAtom::typeConstant:
    return llvm::ELF::SHT_PROGBITS;

  case DefinedAtom::typeZeroFill:
   return llvm::ELF::SHT_NOBITS;

  // Case to handle section types
  // Symtab, String Table ...
  default:
   return _contentType;
  }
}

/// \brief convert the segment type to a String for diagnostics
///        and printing purposes
template<class ELFT>
StringRef 
Section<ELFT>::segmentKindToStr() const {
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
void Section<ELFT>::write(ELFWriter *writer, llvm::FileOutputBuffer &buffer) {
  uint8_t *chunkBuffer = buffer.getBufferStart();
  for (auto &ai : _atoms) {
    const DefinedAtom *definedAtom = cast<DefinedAtom>(ai._atom);
    if (definedAtom->contentType() == DefinedAtom::typeZeroFill)
      continue;
    // Copy raw content of atom to file buffer.
    llvm::ArrayRef<uint8_t> content = definedAtom->rawContent();
    uint64_t contentSize = content.size();
    if (contentSize == 0)
      continue;
    uint8_t *atomContent = chunkBuffer + ai._fileOffset;
    std::copy_n(content.data(), contentSize, atomContent);
    for (const auto ref : *definedAtom) {
      uint32_t offset = ref->offsetInAtom();
      uint64_t targetAddress = 0;
      assert(ref->target() != nullptr && "Found the target to be NULL");
      targetAddress = writer->addressOfAtom(ref->target());
      uint64_t fixupAddress = writer->addressOfAtom(ai._atom) + offset;
      // apply the relocation
      writer->kindHandler()->applyFixup(ref->kind(),
                                        ref->addend(),
                                        &atomContent[offset],
                                        fixupAddress,
                                        targetAddress);
    }
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

  MergedSections(llvm::StringRef name);

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

  inline llvm::StringRef name() const { return _name; }

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
  llvm::StringRef _name;
  bool _hasSegment;
  uint64_t _ordinal;
  int64_t _flags;
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
    _link = section->link();
    _shInfo = section->shinfo();
    _entSize = section->entsize();
    _type = section->type();
    if (_flags < section->flags())
      _flags = section->flags();
  }
  _kind = c->kind();
  _sections.push_back(c);
}

/// \brief The class represents the ELF String Table
template<class ELFT>
class ELFStringTable : public Section<ELFT> {
public:
  ELFStringTable(const char *str, int32_t order);

  static inline bool classof(const Chunk<ELFT> *c) {
    return c->kind() == Section<ELFT>::K_StringTable;
  }

  uint64_t addString(const llvm::StringRef symname);

  void write(ELFWriter *writer, llvm::FileOutputBuffer &buffer);

  inline void finalize() { }

private:
  std::vector<llvm::StringRef> _strings;
};

template<class ELFT>
ELFStringTable<ELFT>::ELFStringTable(const char *str, 
                                     int32_t order)
  : Section<ELFT>(
      str,
      llvm::ELF::SHT_STRTAB,
      DefinedAtom::perm___,
      order,
      Section<ELFT>::K_StringTable) {
  // the string table has a NULL entry for which
  // add an empty string
  _strings.push_back("");
  this->_fsize = 1;
  this->_align2 = 1;
  this->setOrder(order);
}

template<class ELFT>
uint64_t
ELFStringTable<ELFT>::addString(const StringRef symname) {
  _strings.push_back(symname);
  uint64_t offset = this->_fsize;
  this->_fsize += symname.size() + 1;
  return offset;
}

template <class ELFT>
void ELFStringTable<ELFT>::write(ELFWriter *writer,
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

/// \brief The ELFSymbolTable class represents the symbol table in a ELF file
template<class ELFT>
class ELFSymbolTable : public Section<ELFT> {
public:
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;

  ELFSymbolTable(const char *str, int32_t order);

  void addSymbol(const Atom *atom, int32_t sectionIndex, uint64_t addr = 0);

  void finalize();

  void write(ELFWriter *writer, llvm::FileOutputBuffer &buffer);

  static inline bool classof(const Chunk<ELFT> *c) {
    return c->kind() == Section<ELFT>::K_SymbolTable;
  }

  inline void setStringSection(ELFStringTable<ELFT> *s) {
    _stringSection = s;
  }

private:
  ELFStringTable<ELFT> *_stringSection;
  std::vector<Elf_Sym*> _symbolTable;
  llvm::BumpPtrAllocator _symbolAllocate;
  int64_t _link;
};

/// ELF Symbol Table 
template<class ELFT>
ELFSymbolTable<ELFT>::ELFSymbolTable(const char *str, 
                                     int32_t order)
  : Section<ELFT>(
      str,
      llvm::ELF::SHT_SYMTAB,
      0,
      order,
      Section<ELFT>::K_SymbolTable) {
  this->setOrder(order);
  Elf_Sym *symbol = new (_symbolAllocate.Allocate<Elf_Sym>()) Elf_Sym;
  memset((void *)symbol, 0, sizeof(Elf_Sym));
  _symbolTable.push_back(symbol);
  this->_entSize = sizeof(Elf_Sym);
  this->_fsize = sizeof(Elf_Sym);
  this->_align2 = sizeof(void *);
}

template<class ELFT>
void 
ELFSymbolTable<ELFT>::addSymbol(const Atom *atom, 
                                int32_t sectionIndex, 
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
    lld::DefinedAtom::ContentType ct;
    switch (ct = da->contentType()){
    case  DefinedAtom::typeCode:
      symbol->st_value = addr;
      type = llvm::ELF::STT_FUNC;
      break;
    case  DefinedAtom::typeData:
    case  DefinedAtom::typeConstant:
      symbol->st_value = addr;
      type = llvm::ELF::STT_OBJECT;
      break;
    case  DefinedAtom::typeZeroFill:
      type = llvm::ELF::STT_OBJECT;
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

template<class ELFT>
void 
ELFSymbolTable<ELFT>::finalize() {
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
  this->_shInfo = shInfo;
  this->setLink(_stringSection->ordinal());
}

template <class ELFT>
void ELFSymbolTable<ELFT>::write(ELFWriter *writer,
                                 llvm::FileOutputBuffer &buffer) {
  uint8_t *chunkBuffer = buffer.getBufferStart();
  uint8_t *dest = chunkBuffer + this->fileOffset();
  for (auto sti : _symbolTable) {
    memcpy(dest, sti, sizeof(Elf_Sym));
    dest += sizeof(Elf_Sym);
  }
}

} // elf
} // lld

#endif //LLD_READER_WRITER_ELF_SECTION_CHUNKS_H_
