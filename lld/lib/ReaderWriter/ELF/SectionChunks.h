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
#include "lld/Core/Parallel.h"
#include "lld/Core/range.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileOutputBuffer.h"

namespace lld {
namespace elf {
template <class> class MergedSections;
using namespace llvm::ELF;
template <class ELFT> class Segment;

/// \brief An ELF section.
template <class ELFT> class Section : public Chunk<ELFT> {
public:
  Section(const ELFLinkingContext &context, StringRef name,
          typename Chunk<ELFT>::Kind k = Chunk<ELFT>::Kind::ELFSection)
      : Chunk<ELFT>(name, k, context), _parent(nullptr), _flags(0), _entSize(0),
        _type(0), _link(0), _info(0), _segmentType(SHT_NULL) {}

  /// \brief Modify the section contents before assigning virtual addresses
  //  or assigning file offsets
  virtual void doPreFlight() {}

  /// \brief Finalize the section contents before writing
  virtual void finalize() {}

  /// \brief Does this section have an output segment.
  virtual bool hasOutputSegment() {
    return false;
  }

  /// Return if the section is a loadable section that occupies memory
  virtual bool isLoadableSection() const { return false; }

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

  /// \brief Return the type of content that the section contains
  virtual int getContentType() const {
    if (_flags & llvm::ELF::SHF_EXECINSTR)
      return Chunk<ELFT>::ContentType::Code;
    else if (_flags & llvm::ELF::SHF_WRITE)
      return Chunk<ELFT>::ContentType::Data;
    else if (_flags & llvm::ELF::SHF_ALLOC)
      return Chunk<ELFT>::ContentType::Code;
    else
      return Chunk<ELFT>::ContentType::Unknown;
  }

  /// \brief convert the segment type to a String for diagnostics and printing
  /// purposes
  StringRef segmentKindToStr() const;

  /// \brief Records the segmentType, that this section belongs to
  void setSegmentType(const Layout::SegmentType segmentType) {
    this->_segmentType = segmentType;
  }

  virtual bool findAtomAddrByName(StringRef, uint64_t &) { return false; }

  void setMergedSection(MergedSections<ELFT> *ms) { _parent = ms; }

  static bool classof(const Chunk<ELFT> *c) {
    return c->kind() == Chunk<ELFT>::Kind::ELFSection ||
           c->kind() == Chunk<ELFT>::Kind::AtomSection;
  }

protected:
  /// \brief MergedSections this Section is a member of, or nullptr.
  MergedSections<ELFT> *_parent;
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
  AtomSection(const ELFLinkingContext &context, StringRef name,
              int32_t contentType, int32_t permissions, int32_t order)
      : Section<ELFT>(context, name, Chunk<ELFT>::Kind::AtomSection),
        _contentType(contentType), _contentPermissions(permissions),
        _isLoadedInMemory(true) {
    this->setOrder(order);

    switch (contentType) {
    case DefinedAtom::typeCode:
    case DefinedAtom::typeDataFast:
    case DefinedAtom::typeData:
    case DefinedAtom::typeConstant:
    case DefinedAtom::typeGOT:
    case DefinedAtom::typeStub:
    case DefinedAtom::typeResolver:
    case DefinedAtom::typeThreadData:
      this->_type = SHT_PROGBITS;
      break;

    case DefinedAtom::typeThreadZeroFill:
    case DefinedAtom::typeZeroFillFast:
    case DefinedAtom::typeZeroFill:
      this->_type = SHT_NOBITS;
      break;

    case DefinedAtom::typeRONote:
    case DefinedAtom::typeRWNote:
      this->_type = SHT_NOTE;
      break;

    case DefinedAtom::typeNoAlloc:
      this->_type = SHT_PROGBITS;
      this->_isLoadedInMemory = false;
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
      if (_contentType == DefinedAtom::typeThreadData ||
          _contentType == DefinedAtom::typeThreadZeroFill)
        this->_flags |= SHF_TLS;
      break;
    case DefinedAtom::permRWX:
      this->_flags = SHF_ALLOC | SHF_WRITE | SHF_EXECINSTR;
      break;
    case DefinedAtom::perm___:
      this->_flags = 0;
      break;
    }
  }

  /// Align the offset to the required modulus defined by the atom alignment
  uint64_t alignOffset(uint64_t offset, DefinedAtom::Alignment &atomAlign);

  /// Return if the section is a loadable section that occupies memory
  virtual bool isLoadableSection() const { return _isLoadedInMemory; }

  // \brief Append an atom to a Section. The atom gets pushed into a vector
  // contains the atom, the atom file offset, the atom virtual address
  // the atom file offset is aligned appropriately as set by the Reader
  virtual const lld::AtomLayout &appendAtom(const Atom *atom);

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

  /// \brief Find the Atom address given a name, this is needed to properly
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

  /// \brief Return the raw flags, we need this to sort segments
  inline int64_t atomflags() const {
    return _contentPermissions;
  }

  /// Atom Iterators
  typedef typename std::vector<lld::AtomLayout *>::iterator atom_iter;

  range<atom_iter> atoms() { return _atoms; }

  virtual void write(ELFWriter *writer, TargetLayout<ELFT> &layout,
                     llvm::FileOutputBuffer &buffer);

  static bool classof(const Chunk<ELFT> *c) {
    return c->kind() == Chunk<ELFT>::Kind::AtomSection;
  }

protected:
  llvm::BumpPtrAllocator _alloc;
  int32_t _contentType;
  int32_t _contentPermissions;
  bool _isLoadedInMemory;
  std::vector<lld::AtomLayout *> _atoms;
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
const lld::AtomLayout &AtomSection<ELFT>::appendAtom(const Atom *atom) {
  Atom::Definition atomType = atom->definition();
  const DefinedAtom *definedAtom = cast<DefinedAtom>(atom);

  DefinedAtom::Alignment atomAlign = definedAtom->alignment();
  uint64_t align2 = 1u << atomAlign.powerOf2;
  // Align the atom to the required modulus/ align the file offset and the
  // memory offset separately this is required so that BSS symbols are handled
  // properly as the BSS symbols only occupy memory size and not file size
  uint64_t fOffset = alignOffset(this->fileSize(), atomAlign);
  uint64_t mOffset = alignOffset(this->memSize(), atomAlign);
  switch (atomType) {
  case Atom::definitionRegular:
    switch(definedAtom->contentType()) {
    case DefinedAtom::typeCode:
    case DefinedAtom::typeConstant:
    case DefinedAtom::typeData:
    case DefinedAtom::typeDataFast:
    case DefinedAtom::typeZeroFillFast:
    case DefinedAtom::typeGOT:
    case DefinedAtom::typeStub:
    case DefinedAtom::typeResolver:
    case DefinedAtom::typeThreadData:
    case DefinedAtom::typeRONote:
    case DefinedAtom::typeRWNote:
      _atoms.push_back(new (_alloc) lld::AtomLayout(atom, fOffset, 0));
      this->_fsize = fOffset + definedAtom->size();
      this->_msize = mOffset + definedAtom->size();
      DEBUG_WITH_TYPE("Section",
                      llvm::dbgs() << "[" << this->name() << " " << this << "] "
                                   << "Adding atom: " << atom->name() << "@"
                                   << fOffset << "\n");
      break;
    case DefinedAtom::typeNoAlloc:
      _atoms.push_back(new (_alloc) lld::AtomLayout(atom, fOffset, 0));
      this->_fsize = fOffset + definedAtom->size();
      DEBUG_WITH_TYPE("Section", llvm::dbgs() << "[" << this->name() << " "
                                              << this << "] "
                                              << "Adding atom: " << atom->name()
                                              << "@" << fOffset << "\n");
      break;
    case DefinedAtom::typeThreadZeroFill:
    case DefinedAtom::typeZeroFill:
      _atoms.push_back(new (_alloc) lld::AtomLayout(atom, mOffset, 0));
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
  // std::max doesn't support uint64_t
  if (this->_align2 < align2)
    this->_align2 = align2;

  return *_atoms.back();
}

/// \brief convert the segment type to a String for diagnostics
///        and printing purposes
template <class ELFT> StringRef Section<ELFT>::segmentKindToStr() const {
  switch(_segmentType) {
  case llvm::ELF::PT_DYNAMIC:
    return "DYNAMIC";
  case llvm::ELF::PT_INTERP:
    return "INTERP";
  case llvm::ELF::PT_LOAD:
    return "LOAD";
  case llvm::ELF::PT_GNU_EH_FRAME:
    return "EH_FRAME";
  case llvm::ELF::PT_GNU_RELRO:
    return "RELRO";
  case llvm::ELF::PT_NOTE:
    return "NOTE";
  case llvm::ELF::PT_NULL:
    return "NULL";
  case llvm::ELF::PT_TLS:
    return "TLS";
  default:
    return "UNKNOWN";
  }
}

/// \brief Write the section and the atom contents to the buffer
template <class ELFT>
void AtomSection<ELFT>::write(ELFWriter *writer, TargetLayout<ELFT> &layout,
                              llvm::FileOutputBuffer &buffer) {
  uint8_t *chunkBuffer = buffer.getBufferStart();
  parallel_for_each(_atoms.begin(), _atoms.end(), [&](lld::AtomLayout * ai) {
    DEBUG_WITH_TYPE("Section",
                    llvm::dbgs() << "Writing atom: " << ai->_atom->name()
                                 << " | " << ai->_fileOffset << "\n");
    const DefinedAtom *definedAtom = cast<DefinedAtom>(ai->_atom);
    if (!definedAtom->occupiesDiskSpace())
      return;
    // Copy raw content of atom to file buffer.
    ArrayRef<uint8_t> content = definedAtom->rawContent();
    uint64_t contentSize = content.size();
    if (contentSize == 0)
      return;
    uint8_t *atomContent = chunkBuffer + ai->_fileOffset;
    std::memcpy(atomContent, content.data(), contentSize);
    const TargetRelocationHandler<ELFT> &relHandler =
        this->_context.template getTargetHandler<ELFT>().getRelocationHandler();
    for (const auto ref : *definedAtom)
      relHandler.applyRelocation(*writer, buffer, *ai, *ref);
  });
}

/// \brief A MergedSections represents a set of sections grouped by the same
/// name. The output file that gets written by the linker has sections grouped
/// by similar names
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

  // Is the section loadable ?
  inline bool isLoadableSection() const { return _isLoadableSection; }

  // Set section Loadable
  inline void setLoadableSection(bool isLoadable) {
    _isLoadableSection = isLoadable;
  }

  void setLink(uint64_t link) { _link = link; }

  void setInfo(uint64_t info) { _shInfo = info; }

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
  bool _isLoadableSection;
  std::vector<Chunk<ELFT> *> _sections;
};

/// MergedSections
template <class ELFT>
MergedSections<ELFT>::MergedSections(StringRef name)
    : _name(name), _hasSegment(false), _ordinal(0), _flags(0), _size(0),
      _memSize(0), _fileOffset(0), _virtualAddr(0), _shInfo(0), _entSize(0),
      _link(0), _align2(0), _kind(0), _type(0), _isLoadableSection(false) {}

template<class ELFT>
void
MergedSections<ELFT>::appendSection(Chunk<ELFT> *c) {
  if (c->align2() > _align2)
    _align2 = c->align2();
  if (const auto section = dyn_cast<Section<ELFT>>(c)) {
    assert(!_link && "Section already has a link!");
    _link = section->getLink();
    _shInfo = section->getInfo();
    _entSize = section->getEntSize();
    _type = section->getType();
    if (_flags < section->getFlags())
      _flags = section->getFlags();
    section->setMergedSection(this);
  }
  _kind = c->kind();
  _sections.push_back(c);
}

/// \brief The class represents the ELF String Table
template<class ELFT>
class StringTable : public Section<ELFT> {
public:
  StringTable(const ELFLinkingContext &, const char *str, int32_t order,
              bool dynamic = false);

  uint64_t addString(StringRef symname);

  virtual void write(ELFWriter *writer, TargetLayout<ELFT> &layout,
                     llvm::FileOutputBuffer &buffer);

  inline void setNumEntries(int64_t numEntries) {
    _stringMap.resize(numEntries);
  }

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
StringTable<ELFT>::StringTable(const ELFLinkingContext &context,
                               const char *str, int32_t order, bool dynamic)
    : Section<ELFT>(context, str) {
  // the string table has a NULL entry for which
  // add an empty string
  _strings.push_back("");
  this->_fsize = 1;
  this->_align2 = 1;
  this->setOrder(order);
  this->_type = SHT_STRTAB;
  if (dynamic) {
    this->_flags = SHF_ALLOC;
    this->_msize = this->_fsize;
  }
}

template <class ELFT> uint64_t StringTable<ELFT>::addString(StringRef symname) {
  if (symname.empty())
    return 0;
  StringMapTIter stringIter = _stringMap.find(symname);
  if (stringIter == _stringMap.end()) {
    _strings.push_back(symname);
    uint64_t offset = this->_fsize;
    this->_fsize += symname.size() + 1;
    if (this->_flags & SHF_ALLOC)
      this->_msize = this->_fsize;
    _stringMap[symname] = offset;
    return offset;
  }
  return stringIter->second;
}

template <class ELFT>
void StringTable<ELFT>::write(ELFWriter *writer, TargetLayout<ELFT> &,
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
  typedef typename llvm::object::ELFDataTypeTypedefHelper<ELFT>::Elf_Addr
      Elf_Addr;

public:
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;

  SymbolTable(const ELFLinkingContext &context, const char *str, int32_t order);

  /// \brief set the number of entries that would exist in the symbol
  /// table for the current link
  void setNumEntries(int64_t numEntries) const {
    if (_stringSection)
      _stringSection->setNumEntries(numEntries);
  }

  /// \brief return number of entries
  std::size_t size() const { return _symbolTable.size(); }

  void addSymbol(const Atom *atom, int32_t sectionIndex, uint64_t addr = 0,
                 const lld::AtomLayout *layout = nullptr);

  /// \brief Get the symbol table index for an Atom. If it's not in the symbol
  /// table, return STN_UNDEF.
  uint32_t getSymbolTableIndex(const Atom *a) const {
    for (size_t i = 0, e = _symbolTable.size(); i < e; ++i)
      if (_symbolTable[i]._atom == a)
        return i;
    return STN_UNDEF;
  }

  virtual void finalize() { finalize(true); }

  virtual void sortSymbols() {
    std::stable_sort(_symbolTable.begin(), _symbolTable.end(),
                     [](const SymbolEntry & A, const SymbolEntry & B) {
      return A._symbol.getBinding() < B._symbol.getBinding();
    });
  }

  virtual void addAbsoluteAtom(Elf_Sym &sym, const AbsoluteAtom *aa,
                               int64_t addr);

  virtual void addDefinedAtom(Elf_Sym &sym, const DefinedAtom *da,
                              int64_t addr);

  virtual void addUndefinedAtom(Elf_Sym &sym, const UndefinedAtom *ua);

  virtual void addSharedLibAtom(Elf_Sym &sym, const SharedLibraryAtom *sla);

  virtual void finalize(bool sort = true);

  virtual void write(ELFWriter *writer, TargetLayout<ELFT> &layout,
                     llvm::FileOutputBuffer &buffer);

  void setStringSection(StringTable<ELFT> *s) { _stringSection = s; }

  StringTable<ELFT> *getStringTable() const { return _stringSection; }

protected:
  struct SymbolEntry {
    SymbolEntry(const Atom *a, const Elf_Sym &sym,
                const lld::AtomLayout *layout)
        : _atom(a), _atomLayout(layout), _symbol(sym) {}

    const Atom *_atom;
    const lld::AtomLayout *_atomLayout;
    Elf_Sym _symbol;
  };

  llvm::BumpPtrAllocator _symbolAllocate;
  StringTable<ELFT> *_stringSection;
  std::vector<SymbolEntry> _symbolTable;
};

/// ELF Symbol Table
template <class ELFT>
SymbolTable<ELFT>::SymbolTable(const ELFLinkingContext &context,
                               const char *str, int32_t order)
    : Section<ELFT>(context, str) {
  this->setOrder(order);
  Elf_Sym symbol;
  std::memset(&symbol, 0, sizeof(Elf_Sym));
  _symbolTable.push_back(SymbolEntry(nullptr, symbol, nullptr));
  this->_entSize = sizeof(Elf_Sym);
  this->_fsize = sizeof(Elf_Sym);
  this->_align2 = sizeof(Elf_Addr);
  this->_type = SHT_SYMTAB;
}

template <class ELFT>
void SymbolTable<ELFT>::addDefinedAtom(Elf_Sym &sym, const DefinedAtom *da,
                                       int64_t addr) {
  unsigned char binding = 0, type = 0;
  sym.st_size = da->size();
  DefinedAtom::ContentType ct;
  switch (ct = da->contentType()) {
  case DefinedAtom::typeCode:
  case DefinedAtom::typeStub:
    sym.st_value = addr;
    type = llvm::ELF::STT_FUNC;
    break;
  case DefinedAtom::typeResolver:
    sym.st_value = addr;
    type = llvm::ELF::STT_GNU_IFUNC;
    break;
  case DefinedAtom::typeDataFast:
  case DefinedAtom::typeData:
  case DefinedAtom::typeConstant:
    sym.st_value = addr;
    type = llvm::ELF::STT_OBJECT;
    break;
  case DefinedAtom::typeGOT:
    sym.st_value = addr;
    type = llvm::ELF::STT_NOTYPE;
    break;
  case DefinedAtom::typeZeroFill:
  case DefinedAtom::typeZeroFillFast:
    type = llvm::ELF::STT_OBJECT;
    sym.st_value = addr;
    break;
  case DefinedAtom::typeThreadData:
  case DefinedAtom::typeThreadZeroFill:
    type = llvm::ELF::STT_TLS;
    sym.st_value = addr;
    break;
  default:
    type = llvm::ELF::STT_NOTYPE;
  }
  if (da->customSectionName() == da->name())
    type = llvm::ELF::STT_SECTION;

  if (da->scope() == DefinedAtom::scopeTranslationUnit)
    binding = llvm::ELF::STB_LOCAL;
  else
    binding = llvm::ELF::STB_GLOBAL;

  sym.setBindingAndType(binding, type);
}

template <class ELFT>
void SymbolTable<ELFT>::addAbsoluteAtom(Elf_Sym &sym, const AbsoluteAtom *aa,
                                        int64_t addr) {
  unsigned char binding = 0, type = 0;
  type = llvm::ELF::STT_OBJECT;
  sym.st_shndx = llvm::ELF::SHN_ABS;
  switch (aa->scope()) {
  case AbsoluteAtom::scopeLinkageUnit:
    sym.st_other = llvm::ELF::STV_HIDDEN;
    binding = llvm::ELF::STB_LOCAL;
    break;
  case AbsoluteAtom::scopeTranslationUnit:
    binding = llvm::ELF::STB_LOCAL;
    break;
  case AbsoluteAtom::scopeGlobal:
    binding = llvm::ELF::STB_GLOBAL;
    break;
  }
  sym.st_value = addr;
  sym.setBindingAndType(binding, type);
}

template <class ELFT>
void SymbolTable<ELFT>::addSharedLibAtom(Elf_Sym &sym,
                                         const SharedLibraryAtom *aa) {
  unsigned char binding = 0, type = 0;
  if (aa->type() == SharedLibraryAtom::Type::Data) {
    type = llvm::ELF::STT_OBJECT;
    sym.st_size = aa->size();
  } else
    type = llvm::ELF::STT_FUNC;
  sym.st_shndx = llvm::ELF::SHN_UNDEF;
  binding = llvm::ELF::STB_GLOBAL;
  sym.setBindingAndType(binding, type);
}

template <class ELFT>
void SymbolTable<ELFT>::addUndefinedAtom(Elf_Sym &sym,
                                         const UndefinedAtom *ua) {
  unsigned char binding = 0, type = 0;
  sym.st_value = 0;
  type = llvm::ELF::STT_NOTYPE;
  if (ua->canBeNull())
    binding = llvm::ELF::STB_WEAK;
  else
    binding = llvm::ELF::STB_GLOBAL;
  sym.setBindingAndType(binding, type);
}

/// Add a symbol to the symbol Table, definedAtoms which get added to the symbol
/// section don't have their virtual addresses set at the time of adding the
/// symbol to the symbol table(Example: dynamic symbols), the addresses needs
/// to be updated in the table before writing the dynamic symbol table
/// information
template <class ELFT>
void SymbolTable<ELFT>::addSymbol(const Atom *atom, int32_t sectionIndex,
                                  uint64_t addr,
                                  const lld::AtomLayout *atomLayout) {
  Elf_Sym symbol;

  if (atom->name().empty())
    return;

  symbol.st_name = _stringSection->addString(atom->name());
  symbol.st_size = 0;
  symbol.st_shndx = sectionIndex;
  symbol.st_value = 0;
  symbol.st_other = llvm::ELF::STV_DEFAULT;

  // Add all the atoms
  if (const DefinedAtom *da = dyn_cast<const DefinedAtom>(atom))
    addDefinedAtom(symbol, da, addr);
  else if (const AbsoluteAtom *aa = dyn_cast<const AbsoluteAtom>(atom))
    addAbsoluteAtom(symbol, aa, addr);
  else if (isa<const SharedLibraryAtom>(atom))
    addSharedLibAtom(symbol, dyn_cast<SharedLibraryAtom>(atom));
  else
    addUndefinedAtom(symbol, dyn_cast<UndefinedAtom>(atom));

  _symbolTable.push_back(SymbolEntry(atom, symbol, atomLayout));
  this->_fsize += sizeof(Elf_Sym);
  if (this->_flags & SHF_ALLOC)
    this->_msize = this->_fsize;
}

template <class ELFT> void SymbolTable<ELFT>::finalize(bool sort) {
  // sh_info should be one greater than last symbol with STB_LOCAL binding
  // we sort the symbol table to keep all local symbols at the beginning
  if (sort)
    sortSymbols();

  uint16_t shInfo = 0;
  for (const auto &i : _symbolTable) {
    if (i._symbol.getBinding() != llvm::ELF::STB_LOCAL)
      break;
    shInfo++;
  }
  this->_info = shInfo;
  this->_link = _stringSection->ordinal();
  if (this->_parent) {
    this->_parent->setInfo(this->_info);
    this->_parent->setLink(this->_link);
  }
}

template <class ELFT>
void SymbolTable<ELFT>::write(ELFWriter *writer, TargetLayout<ELFT> &,
                              llvm::FileOutputBuffer &buffer) {
  uint8_t *chunkBuffer = buffer.getBufferStart();
  uint8_t *dest = chunkBuffer + this->fileOffset();
  for (const auto &sti : _symbolTable) {
    memcpy(dest, &sti._symbol, sizeof(Elf_Sym));
    dest += sizeof(Elf_Sym);
  }
}

template <class ELFT> class HashSection;

template <class ELFT> class DynamicSymbolTable : public SymbolTable<ELFT> {
public:
  DynamicSymbolTable(const ELFLinkingContext &context,
                     TargetLayout<ELFT> &layout, const char *str, int32_t order)
      : SymbolTable<ELFT>(context, str, order), _hashTable(nullptr),
        _layout(layout) {
    this->_type = SHT_DYNSYM;
    this->_flags = SHF_ALLOC;
    this->_msize = this->_fsize;
  }

  // Set the dynamic hash table for symbols to be added into
  void setHashTable(HashSection<ELFT> *hashTable) { _hashTable = hashTable; }

  // Add all the dynamic symbos to the hash table
  void addSymbolsToHashTable() {
    int index = 0;
    for (auto &ste : this->_symbolTable) {
      if (!ste._atom)
        _hashTable->addSymbol("", index);
      else
        _hashTable->addSymbol(ste._atom->name(), index);
      ++index;
    }
  }

  virtual void finalize() {
    // Defined symbols which have been added into the dynamic symbol table
    // don't have their addresses known until addresses have been assigned
    // so let's update the symbol values after they have got assigned
    for (auto &ste: this->_symbolTable) {
      const lld::AtomLayout *atomLayout = ste._atomLayout;
      if (!atomLayout)
        continue;
      ste._symbol.st_value = atomLayout->_virtualAddr;
    }

    // Don't sort the symbols
    SymbolTable<ELFT>::finalize(false);
  }

protected:
  HashSection<ELFT> *_hashTable;
  TargetLayout<ELFT> &_layout;
};

template <class ELFT> class RelocationTable : public Section<ELFT> {
public:
  typedef llvm::object::Elf_Rel_Impl<ELFT, false> Elf_Rel;
  typedef llvm::object::Elf_Rel_Impl<ELFT, true> Elf_Rela;

  RelocationTable(const ELFLinkingContext &context, StringRef str,
                  int32_t order)
      : Section<ELFT>(context, str), _symbolTable(nullptr) {
    this->setOrder(order);
    this->_flags = SHF_ALLOC;
    // Set the alignment properly depending on the target architecture
    if (context.is64Bits())
      this->_align2 = 8;
    else
      this->_align2 = 4;
    if (context.isRelaOutputFormat()) {
      this->_entSize = sizeof(Elf_Rela);
      this->_type = SHT_RELA;
    } else {
      this->_entSize = sizeof(Elf_Rel);
      this->_type = SHT_REL;
    }
  }

  /// \returns the index of the relocation added.
  uint32_t addRelocation(const DefinedAtom &da, const Reference &r) {
    _relocs.emplace_back(&da, &r);
    this->_fsize = _relocs.size() * this->_entSize;
    this->_msize = this->_fsize;
    return _relocs.size() - 1;
  }

  bool getRelocationIndex(const Reference &r, uint32_t &res) {
    auto rel = std::find_if(
        _relocs.begin(), _relocs.end(),
        [&](const std::pair<const DefinedAtom *, const Reference *> &p) {
      if (p.second == &r)
        return true;
      return false;
    });
    if (rel == _relocs.end())
      return false;
    res = std::distance(_relocs.begin(), rel);
    return true;
  }

  void setSymbolTable(const DynamicSymbolTable<ELFT> *symbolTable) {
    _symbolTable = symbolTable;
  }

  virtual void finalize() {
    this->_link = _symbolTable ? _symbolTable->ordinal() : 0;
    if (this->_parent)
      this->_parent->setLink(this->_link);
  }

  virtual void write(ELFWriter *writer, TargetLayout<ELFT> &layout,
                     llvm::FileOutputBuffer &buffer) {
    uint8_t *chunkBuffer = buffer.getBufferStart();
    uint8_t *dest = chunkBuffer + this->fileOffset();
    for (const auto &rel : _relocs) {
      if (this->_context.isRelaOutputFormat())
        writeRela(writer, *reinterpret_cast<Elf_Rela *>(dest), *rel.first,
                  *rel.second);
      else
        writeRel(writer, *reinterpret_cast<Elf_Rel *>(dest), *rel.first,
                 *rel.second);
      dest += this->_entSize;
    }
  }

private:
  std::vector<std::pair<const DefinedAtom *, const Reference *> > _relocs;
  const DynamicSymbolTable<ELFT> *_symbolTable;

  void writeRela(ELFWriter *writer, Elf_Rela &r, const DefinedAtom &atom,
                 const Reference &ref) {
    uint32_t index =
        _symbolTable ? _symbolTable->getSymbolTableIndex(ref.target())
                     : (uint32_t)STN_UNDEF;
    r.setSymbolAndType(index, ref.kindValue());
    r.r_offset = writer->addressOfAtom(&atom) + ref.offsetInAtom();
    r.r_addend = 0;
    // The addend is used only by relative relocations
    if (this->_context.isRelativeReloc(ref))
      r.r_addend = writer->addressOfAtom(ref.target()) + ref.addend();
    DEBUG_WITH_TYPE("ELFRelocationTable",
                    llvm::dbgs() << ref.kindValue() << " relocation at "
                                 << atom.name() << "@" << r.r_offset << " to "
                                 << ref.target()->name() << "@" << r.r_addend
                                 << "\n";);
  }

  void writeRel(ELFWriter *writer, Elf_Rel &r, const DefinedAtom &atom,
                const Reference &ref) {
    uint32_t index =
        _symbolTable ? _symbolTable->getSymbolTableIndex(ref.target())
                     : (uint32_t)STN_UNDEF;
    r.setSymbolAndType(index, ref.kindValue());
    r.r_offset = writer->addressOfAtom(&atom) + ref.offsetInAtom();
    DEBUG_WITH_TYPE("ELFRelocationTable",
                    llvm::dbgs() << ref.kindValue() << " relocation at "
                                 << atom.name() << "@" << r.r_offset << " to "
                                 << ref.target()->name() << "\n";);
  }
};

template <class ELFT> class HashSection;

template <class ELFT> class DynamicTable : public Section<ELFT> {
public:
  typedef llvm::object::Elf_Dyn_Impl<ELFT> Elf_Dyn;
  typedef std::vector<Elf_Dyn> EntriesT;

  DynamicTable(const ELFLinkingContext &context, TargetLayout<ELFT> &layout,
               StringRef str, int32_t order)
      : Section<ELFT>(context, str), _layout(layout) {
    this->setOrder(order);
    this->_entSize = sizeof(Elf_Dyn);
    this->_align2 = llvm::alignOf<Elf_Dyn>();
    // Reserve space for the DT_NULL entry.
    this->_fsize = sizeof(Elf_Dyn);
    this->_msize = sizeof(Elf_Dyn);
    this->_type = SHT_DYNAMIC;
    this->_flags = SHF_ALLOC;
  }

  range<typename EntriesT::iterator> entries() { return _entries; }

  /// \returns the index of the entry.
  std::size_t addEntry(Elf_Dyn e) {
    _entries.push_back(e);
    this->_fsize = (_entries.size() * sizeof(Elf_Dyn)) + sizeof(Elf_Dyn);
    this->_msize = this->_fsize;
    return _entries.size() - 1;
  }

  void write(ELFWriter *writer, TargetLayout<ELFT> &layout,
             llvm::FileOutputBuffer &buffer) {
    uint8_t *chunkBuffer = buffer.getBufferStart();
    uint8_t *dest = chunkBuffer + this->fileOffset();
    // Add the null entry.
    Elf_Dyn d;
    d.d_tag = 0;
    d.d_un.d_val = 0;
    _entries.push_back(d);
    std::memcpy(dest, _entries.data(), this->_fsize);
  }

  virtual void createDefaultEntries() {
    bool isRela = this->_context.isRelaOutputFormat();

    Elf_Dyn dyn;
    dyn.d_un.d_val = 0;

    dyn.d_tag = DT_HASH;
    _dt_hash = addEntry(dyn);
    dyn.d_tag = DT_STRTAB;
    _dt_strtab = addEntry(dyn);
    dyn.d_tag = DT_SYMTAB;
    _dt_symtab = addEntry(dyn);
    dyn.d_tag = DT_STRSZ;
    _dt_strsz = addEntry(dyn);
    dyn.d_tag = DT_SYMENT;
    _dt_syment = addEntry(dyn);
    dyn.d_tag = DT_FINI_ARRAY;
    _dt_fini_array = addEntry(dyn);
    dyn.d_tag = DT_FINI_ARRAYSZ;
    _dt_fini_arraysz = addEntry(dyn);
    if (_layout.hasDynamicRelocationTable()) {
      dyn.d_tag = isRela ? DT_RELA : DT_REL;
      _dt_rela = addEntry(dyn);
      dyn.d_tag = isRela ? DT_RELASZ : DT_RELSZ;
      _dt_relasz = addEntry(dyn);
      dyn.d_tag = isRela ? DT_RELAENT : DT_RELENT;
      _dt_relaent = addEntry(dyn);
    }
    if (_layout.hasPLTRelocationTable()) {
      dyn.d_tag = DT_PLTRELSZ;
      _dt_pltrelsz = addEntry(dyn);
      dyn.d_tag = getGotPltTag();
      _dt_pltgot = addEntry(dyn);
      dyn.d_tag = DT_PLTREL;
      dyn.d_un.d_val = isRela ? DT_RELA : DT_REL;
      _dt_pltrel = addEntry(dyn);
      dyn.d_un.d_val = 0;
      dyn.d_tag = DT_JMPREL;
      _dt_jmprel = addEntry(dyn);
    }
  }

  /// \brief Dynamic table tag for .got.plt section referencing.
  /// Usually but not always targets use DT_PLTGOT for that.
  virtual int64_t getGotPltTag() { return DT_PLTGOT; }

  virtual void finalize() {
    StringTable<ELFT> *dynamicStringTable =
        _dynamicSymbolTable->getStringTable();
    this->_link = dynamicStringTable->ordinal();
    if (this->_parent) {
      this->_parent->setInfo(this->_info);
      this->_parent->setLink(this->_link);
    }
  }

  void setSymbolTable(DynamicSymbolTable<ELFT> *dynsym) {
    _dynamicSymbolTable = dynsym;
  }

  const DynamicSymbolTable<ELFT> *getSymbolTable() const {
    return _dynamicSymbolTable;
  }

  void setHashTable(HashSection<ELFT> *hsh) { _hashTable = hsh; }

  virtual void updateDynamicTable() {
    StringTable<ELFT> *dynamicStringTable =
        _dynamicSymbolTable->getStringTable();
    _entries[_dt_hash].d_un.d_val = _hashTable->virtualAddr();
    _entries[_dt_strtab].d_un.d_val = dynamicStringTable->virtualAddr();
    _entries[_dt_symtab].d_un.d_val = _dynamicSymbolTable->virtualAddr();
    _entries[_dt_strsz].d_un.d_val = dynamicStringTable->memSize();
    _entries[_dt_syment].d_un.d_val = _dynamicSymbolTable->getEntSize();
    auto finiArray = _layout.findOutputSection(".fini_array");
    if (finiArray) {
      _entries[_dt_fini_array].d_un.d_val = finiArray->virtualAddr();
      _entries[_dt_fini_arraysz].d_un.d_val = finiArray->memSize();
    }
    if (_layout.hasDynamicRelocationTable()) {
      auto relaTbl = _layout.getDynamicRelocationTable();
      _entries[_dt_rela].d_un.d_val = relaTbl->virtualAddr();
      _entries[_dt_relasz].d_un.d_val = relaTbl->memSize();
      _entries[_dt_relaent].d_un.d_val = relaTbl->getEntSize();
    }
    if (_layout.hasPLTRelocationTable()) {
      auto relaTbl = _layout.getPLTRelocationTable();
      _entries[_dt_jmprel].d_un.d_val = relaTbl->virtualAddr();
      _entries[_dt_pltrelsz].d_un.d_val = relaTbl->memSize();
      auto gotplt = _layout.findOutputSection(".got.plt");
      _entries[_dt_pltgot].d_un.d_val = gotplt->virtualAddr();
    }
  }

protected:
  EntriesT _entries;

private:
  std::size_t _dt_hash;
  std::size_t _dt_strtab;
  std::size_t _dt_symtab;
  std::size_t _dt_rela;
  std::size_t _dt_relasz;
  std::size_t _dt_relaent;
  std::size_t _dt_strsz;
  std::size_t _dt_syment;
  std::size_t _dt_pltrelsz;
  std::size_t _dt_pltgot;
  std::size_t _dt_pltrel;
  std::size_t _dt_jmprel;
  std::size_t _dt_fini_array;
  std::size_t _dt_fini_arraysz;
  TargetLayout<ELFT> &_layout;
  DynamicSymbolTable<ELFT> *_dynamicSymbolTable;
  HashSection<ELFT> *_hashTable;
};

template <class ELFT> class InterpSection : public Section<ELFT> {
public:
  InterpSection(const ELFLinkingContext &context, StringRef str, int32_t order,
                StringRef interp)
      : Section<ELFT>(context, str), _interp(interp) {
    this->setOrder(order);
    this->_align2 = 1;
    // + 1 for null term.
    this->_fsize = interp.size() + 1;
    this->_msize = this->_fsize;
    this->_type = SHT_PROGBITS;
    this->_flags = SHF_ALLOC;
  }

  void write(ELFWriter *writer, TargetLayout<ELFT> &layout,
             llvm::FileOutputBuffer &buffer) {
    uint8_t *chunkBuffer = buffer.getBufferStart();
    uint8_t *dest = chunkBuffer + this->fileOffset();
    std::memcpy(dest, _interp.data(), _interp.size());
  }

private:
  StringRef _interp;
};

/// The hash table in the dynamic linker is organized into
///
///     [ nbuckets              ]
///     [ nchains               ]
///     [ buckets[0]            ]
///     .........................
///     [ buckets[nbuckets-1]   ]
///     [ chains[0]             ]
///     .........................
///     [ chains[nchains - 1]   ]
///
/// nbuckets - total number of hash buckets
/// nchains is equal to the number of dynamic symbols.
///
/// The symbol is searched by the dynamic linker using the below approach.
///  * Calculate the hash of the symbol that needs to be searched
///  * Take the value from the buckets[hash % nbuckets] as the index of symbol
///  * Compare the symbol's name, if true return, if false, look through the
///  * array since there was a collision

template <class ELFT> class HashSection : public Section<ELFT> {
  struct SymbolTableEntry {
    StringRef _name;
    uint32_t _index;
  };

public:
  HashSection(const ELFLinkingContext &context, StringRef name, int32_t order)
      : Section<ELFT>(context, name), _symbolTable(nullptr) {
    this->setOrder(order);
    this->_entSize = 4;
    this->_type = SHT_HASH;
    this->_flags = SHF_ALLOC;
    // Set the alignment properly depending on the target architecture
    if (context.is64Bits())
      this->_align2 = 8;
    else
      this->_align2 = 4;
    this->_fsize = 0;
    this->_msize = 0;
  }

  /// \brief add the dynamic symbol into the table so that the
  /// hash could be calculated
  void addSymbol(StringRef name, uint32_t index) {
    SymbolTableEntry ste;
    ste._name = name;
    ste._index = index;
    _entries.push_back(ste);
  }

  /// \brief Set the dynamic symbol table
  void setSymbolTable(const DynamicSymbolTable<ELFT> *symbolTable) {
    _symbolTable = symbolTable;
  }

  // The size of the section has to be determined so that fileoffsets
  // may be properly assigned. Let's calculate the buckets and the chains
  // and fill the chains and the buckets hash table used by the dynamic
  // linker and update the filesize and memory size accordingly
  virtual void doPreFlight() {
    // The number of buckets to use for a certain number of symbols.
    // If there are less than 3 symbols, 1 bucket will be used. If
    // there are less than 17 symbols, 3 buckets will be used, and so
    // forth. The bucket numbers are defined by GNU ld. We use the
    // same rules here so we generate hash sections with the same
    // size as those generated by GNU ld.
    uint32_t hashBuckets[] = { 1, 3, 17, 37, 67, 97, 131, 197, 263, 521, 1031,
                               2053, 4099, 8209, 16411, 32771, 65537, 131101,
                               262147 };
    int hashBucketsCount = sizeof(hashBuckets) / sizeof(uint32_t);

    unsigned int bucketsCount = 0;
    unsigned int dynSymCount = _entries.size();

    // Get the number of buckes that we want to use
    for (int i = 0; i < hashBucketsCount; ++i) {
      if (dynSymCount < hashBuckets[i])
        break;
      bucketsCount = hashBuckets[i];
    }
    _buckets.resize(bucketsCount);
    _chains.resize(_entries.size());

    // Create the hash table for the dynamic linker
    for (auto ai : _entries) {
      unsigned int dynsymIndex = ai._index;
      unsigned int bucketpos = llvm::object::elf_hash(ai._name) % bucketsCount;
      _chains[dynsymIndex] = _buckets[bucketpos];
      _buckets[bucketpos] = dynsymIndex;
    }

    this->_fsize = (2 + _chains.size() + _buckets.size()) * sizeof(uint32_t);
    this->_msize = this->_fsize;
  }

  virtual void finalize() {
    this->_link = _symbolTable ? _symbolTable->ordinal() : 0;
    if (this->_parent)
      this->_parent->setLink(this->_link);
  }

  virtual void write(ELFWriter *writer, TargetLayout<ELFT> &layout,
                     llvm::FileOutputBuffer &buffer) {
    uint8_t *chunkBuffer = buffer.getBufferStart();
    uint8_t *dest = chunkBuffer + this->fileOffset();
    uint32_t bucketChainCounts[2];
    bucketChainCounts[0] = _buckets.size();
    bucketChainCounts[1] = _chains.size();
    std::memcpy(dest, (char *)bucketChainCounts, sizeof(bucketChainCounts));
    dest += sizeof(bucketChainCounts);
    // write bucket values
    for (auto bi : _buckets) {
      uint32_t val = (bi);
      std::memcpy(dest, &val, sizeof(uint32_t));
      dest += sizeof(uint32_t);
    }
    // write chain values
    for (auto ci : _chains) {
      uint32_t val = (ci);
      std::memcpy(dest, &val, sizeof(uint32_t));
      dest += sizeof(uint32_t);
    }
  }

private:
  std::vector<SymbolTableEntry> _entries;
  std::vector<uint32_t> _buckets;
  std::vector<uint32_t> _chains;
  const DynamicSymbolTable<ELFT> *_symbolTable;
};

template <class ELFT> class EHFrameHeader : public Section<ELFT> {
public:
  EHFrameHeader(const ELFLinkingContext &context, StringRef name,
                TargetLayout<ELFT> &layout, int32_t order)
      : Section<ELFT>(context, name), _layout(layout) {
    this->setOrder(order);
    this->_entSize = 0;
    this->_type = SHT_PROGBITS;
    this->_flags = SHF_ALLOC;
    // Set the alignment properly depending on the target architecture
    if (context.is64Bits())
      this->_align2 = 8;
    else
      this->_align2 = 4;
    // Minimum size for empty .eh_frame_hdr.
    this->_fsize = 1 + 1 + 1 + 1 + 4;
    this->_msize = this->_fsize;
  }

  virtual void doPreFlight() override {
    // TODO: Generate a proper binary search table.
  }

  virtual void finalize() override {
    MergedSections<ELFT> *s = _layout.findOutputSection(".eh_frame");
    _ehFrameAddr = s ? s->virtualAddr() : 0;
  }

  virtual void write(ELFWriter *writer, TargetLayout<ELFT> &layout,
                     llvm::FileOutputBuffer &buffer) override {
    uint8_t *chunkBuffer = buffer.getBufferStart();
    uint8_t *dest = chunkBuffer + this->fileOffset();
    int pos = 0;
    dest[pos++] = 1; // version
    dest[pos++] = llvm::dwarf::DW_EH_PE_udata4; // eh_frame_ptr_enc
    dest[pos++] = llvm::dwarf::DW_EH_PE_omit; // fde_count_enc
    dest[pos++] = llvm::dwarf::DW_EH_PE_omit; // table_enc
    *reinterpret_cast<typename llvm::object::ELFFile<ELFT>::Elf_Word *>(
         dest + pos) = (uint32_t)_ehFrameAddr;
  }

private:
  uint64_t _ehFrameAddr;
  TargetLayout<ELFT> &_layout;
};
} // end namespace elf
} // end namespace lld

#endif
