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
#include "TargetHandler.h"
#include "Writer.h"
#include "lld/Core/DefinedAtom.h"
#include "lld/Core/range.h"
#include "lld/ReaderWriter/AtomLayout.h"
#include "lld/ReaderWriter/ELFLinkingContext.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileOutputBuffer.h"
#include <memory>
#include <mutex>

namespace lld {
namespace elf {
template <class> class OutputSection;
using namespace llvm::ELF;
template <class ELFT> class Segment;
template <class ELFT> class TargetLayout;

/// \brief An ELF section.
template <class ELFT> class Section : public Chunk<ELFT> {
public:
  Section(const ELFLinkingContext &ctx, StringRef sectionName,
          StringRef chunkName,
          typename Chunk<ELFT>::Kind k = Chunk<ELFT>::Kind::ELFSection);

  /// \brief Modify the section contents before assigning virtual addresses
  //  or assigning file offsets

  /// \brief Finalize the section contents before writing

  /// \brief Does this section have an output segment.
  virtual bool hasOutputSegment() const { return false; }

  /// Return if the section is a loadable section that occupies memory
  virtual bool isLoadableSection() const { return false; }

  /// \brief Assign file offsets starting at offset.
  virtual void assignFileOffsets(uint64_t offset) {}

  /// \brief Assign virtual addresses starting at addr.
  virtual void assignVirtualAddress(uint64_t addr) {}

  uint64_t getFlags() const { return _flags; }
  uint64_t getEntSize() const { return _entSize; }
  uint32_t getType() const { return _type; }
  uint32_t getLink() const { return _link; }
  uint32_t getInfo() const { return _info; }

  typename TargetLayout<ELFT>::SegmentType getSegmentType() const {
    return _segmentType;
  }

  /// \brief Return the type of content that the section contains
  int getContentType() const override;

  /// \brief convert the segment type to a String for diagnostics and printing
  /// purposes
  virtual StringRef segmentKindToStr() const;

  /// \brief Records the segmentType, that this section belongs to
  void
  setSegmentType(const typename TargetLayout<ELFT>::SegmentType segmentType) {
    this->_segmentType = segmentType;
  }

  virtual const AtomLayout *findAtomLayoutByName(StringRef) const {
    return nullptr;
  }

  void setOutputSection(OutputSection<ELFT> *os, bool isFirst = false) {
    _outputSection = os;
    _isFirstSectionInOutputSection = isFirst;
  }

  static bool classof(const Chunk<ELFT> *c) {
    return c->kind() == Chunk<ELFT>::Kind::ELFSection ||
           c->kind() == Chunk<ELFT>::Kind::AtomSection;
  }

  uint64_t alignment() const override {
    return _isFirstSectionInOutputSection ? _outputSection->alignment()
                                          : this->_alignment;
  }

  virtual StringRef inputSectionName() const { return _inputSectionName; }

  virtual StringRef outputSectionName() const { return _outputSectionName; }

  virtual void setOutputSectionName(StringRef outputSectionName) {
    _outputSectionName = outputSectionName;
  }

  void setArchiveNameOrPath(StringRef name) { _archivePath = name; }

  void setMemberNameOrPath(StringRef name) { _memberPath = name; }

  StringRef archivePath() { return _archivePath; }

  StringRef memberPath() { return _memberPath; }

protected:
  /// \brief OutputSection this Section is a member of, or nullptr.
  OutputSection<ELFT> *_outputSection = nullptr;
  /// \brief ELF SHF_* flags.
  uint64_t _flags = 0;
  /// \brief The size of each entity.
  uint64_t _entSize = 0;
  /// \brief ELF SHT_* type.
  uint32_t _type = 0;
  /// \brief sh_link field.
  uint32_t _link = 0;
  /// \brief the sh_info field.
  uint32_t _info = 0;
  /// \brief Is this the first section in the output section.
  bool _isFirstSectionInOutputSection = false;
  /// \brief the output ELF segment type of this section.
  typename TargetLayout<ELFT>::SegmentType _segmentType = SHT_NULL;
  /// \brief Input section name.
  StringRef _inputSectionName;
  /// \brief Output section name.
  StringRef _outputSectionName;
  StringRef _archivePath;
  StringRef _memberPath;
};

/// \brief A section containing atoms.
template <class ELFT> class AtomSection : public Section<ELFT> {
public:
  AtomSection(const ELFLinkingContext &ctx, StringRef sectionName,
              int32_t contentType, int32_t permissions, int32_t order);

  /// Align the offset to the required modulus defined by the atom alignment
  uint64_t alignOffset(uint64_t offset, DefinedAtom::Alignment &atomAlign);

  /// Return if the section is a loadable section that occupies memory
  bool isLoadableSection() const override { return _isLoadedInMemory; }

  // \brief Append an atom to a Section. The atom gets pushed into a vector
  // contains the atom, the atom file offset, the atom virtual address
  // the atom file offset is aligned appropriately as set by the Reader
  virtual const AtomLayout *appendAtom(const Atom *atom);

  /// \brief Set the virtual address of each Atom in the Section. This
  /// routine gets called after the linker fixes up the virtual address
  /// of the section
  virtual void assignVirtualAddress(uint64_t addr) override;

  /// \brief Set the file offset of each Atom in the section. This routine
  /// gets called after the linker fixes up the section offset
  void assignFileOffsets(uint64_t offset) override;

  /// \brief Find the Atom address given a name, this is needed to properly
  ///  apply relocation. The section class calls this to find the atom address
  ///  to fix the relocation
  const AtomLayout *findAtomLayoutByName(StringRef name) const override;

  /// \brief Return the raw flags, we need this to sort segments
  int64_t atomflags() const { return _contentPermissions; }

  /// Atom Iterators
  typedef typename std::vector<AtomLayout *>::iterator atom_iter;

  range<atom_iter> atoms() { return _atoms; }

  void write(ELFWriter *writer, TargetLayout<ELFT> &layout,
             llvm::FileOutputBuffer &buffer) override;

  static bool classof(const Chunk<ELFT> *c) {
    return c->kind() == Chunk<ELFT>::Kind::AtomSection;
  }

protected:
  llvm::BumpPtrAllocator _alloc;
  int32_t _contentType;
  int32_t _contentPermissions;
  bool _isLoadedInMemory = true;
  std::vector<AtomLayout *> _atoms;
  mutable std::mutex _outputMutex;

  void printError(const std::string &errorStr, const AtomLayout &atom,
                  const Reference &ref) const;
};

/// \brief A OutputSection represents a set of sections grouped by the same
/// name. The output file that gets written by the linker has sections grouped
/// by similar names
template <class ELFT> class OutputSection {
public:
  // Iterators
  typedef typename std::vector<Section<ELFT> *>::iterator SectionIter;

  OutputSection(StringRef name) : _name(name) {}

  // Appends a section into the list of sections that are part of this Output
  // Section
  void appendSection(Section<ELFT> *c);

  // Set the OutputSection is associated with a segment
  void setHasSegment() { _hasSegment = true; }

  /// Sets the ordinal
  void setOrdinal(uint64_t ordinal) { _ordinal = ordinal; }

  /// Sets the Memory size
  void setMemSize(uint64_t memsz) { _memSize = memsz; }

  /// Sets the size fo the output Section.
  void setSize(uint64_t fsiz) { _size = fsiz; }

  // The offset of the first section contained in the output section is
  // contained here.
  void setFileOffset(uint64_t foffset) { _fileOffset = foffset; }

  // Sets the starting address of the section
  void setAddr(uint64_t addr) { _virtualAddr = addr; }

  // Is the section loadable?
  bool isLoadableSection() const { return _isLoadableSection; }

  // Set section Loadable
  void setLoadableSection(bool isLoadable) {
    _isLoadableSection = isLoadable;
  }

  void setLink(uint64_t link) { _link = link; }
  void setInfo(uint64_t info) { _shInfo = info; }
  void setFlag(uint64_t flags) { _flags = flags; }
  void setType(int64_t type) { _type = type; }
  range<SectionIter> sections() { return _sections; }

  // The below functions returns the properties of the OutputSection.
  bool hasSegment() const { return _hasSegment; }
  StringRef name() const { return _name; }
  int64_t shinfo() const { return _shInfo; }
  uint64_t alignment() const { return _alignment; }
  int64_t link() const { return _link; }
  int64_t type() const { return _type; }
  uint64_t virtualAddr() const { return _virtualAddr; }
  int64_t ordinal() const { return _ordinal; }
  int64_t kind() const { return _kind; }
  uint64_t fileSize() const { return _size; }
  int64_t entsize() const { return _entSize; }
  uint64_t fileOffset() const { return _fileOffset; }
  uint64_t flags() const { return _flags; }
  uint64_t memSize() { return _memSize; }

private:
  StringRef _name;
  bool _hasSegment = false;
  uint64_t _ordinal = 0;
  uint64_t _flags = 0;
  uint64_t _size = 0;
  uint64_t _memSize = 0;
  uint64_t _fileOffset = 0;
  uint64_t _virtualAddr = 0;
  int64_t _shInfo = 0;
  int64_t _entSize = 0;
  int64_t _link = 0;
  uint64_t _alignment = 1;
  int64_t _kind = 0;
  int64_t _type = 0;
  bool _isLoadableSection = false;
  std::vector<Section<ELFT> *> _sections;
};

/// \brief The class represents the ELF String Table
template <class ELFT> class StringTable : public Section<ELFT> {
public:
  StringTable(const ELFLinkingContext &, const char *str, int32_t order,
              bool dynamic = false);

  uint64_t addString(StringRef symname);

  void write(ELFWriter *writer, TargetLayout<ELFT> &layout,
             llvm::FileOutputBuffer &buffer) override;

  void setNumEntries(int64_t numEntries) { _stringMap.resize(numEntries); }

private:
  std::vector<StringRef> _strings;

  struct StringRefMappingInfo {
    static StringRef getEmptyKey() { return StringRef(); }
    static StringRef getTombstoneKey() { return StringRef(" ", 1); }
    static unsigned getHashValue(StringRef const val) {
      return llvm::HashString(val);
    }
    static bool isEqual(StringRef const lhs, StringRef const rhs) {
      return lhs.equals(rhs);
    }
  };
  typedef typename llvm::DenseMap<StringRef, uint64_t, StringRefMappingInfo>
      StringMapT;
  typedef typename StringMapT::iterator StringMapTIter;
  StringMapT _stringMap;
};

/// \brief The SymbolTable class represents the symbol table in a ELF file
template <class ELFT> class SymbolTable : public Section<ELFT> {
  typedef
      typename llvm::object::ELFDataTypeTypedefHelper<ELFT>::Elf_Addr Elf_Addr;

public:
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;

  SymbolTable(const ELFLinkingContext &ctx, const char *str, int32_t order);

  /// \brief set the number of entries that would exist in the symbol
  /// table for the current link
  void setNumEntries(int64_t numEntries) const {
    if (_stringSection)
      _stringSection->setNumEntries(numEntries);
  }

  /// \brief return number of entries
  std::size_t size() const { return _symbolTable.size(); }

  void addSymbol(const Atom *atom, int32_t sectionIndex, uint64_t addr = 0,
                 const AtomLayout *layout = nullptr);

  /// \brief Get the symbol table index for an Atom. If it's not in the symbol
  /// table, return STN_UNDEF.
  uint32_t getSymbolTableIndex(const Atom *a) const {
    for (size_t i = 0, e = _symbolTable.size(); i < e; ++i)
      if (_symbolTable[i]._atom == a)
        return i;
    return STN_UNDEF;
  }

  void finalize() override { finalize(true); }

  virtual void sortSymbols() {
    std::stable_sort(_symbolTable.begin(), _symbolTable.end(),
                     [](const SymbolEntry &A, const SymbolEntry &B) {
                       return A._symbol.getBinding() < B._symbol.getBinding();
                     });
  }

  virtual void addAbsoluteAtom(Elf_Sym &sym, const AbsoluteAtom *aa,
                               int64_t addr);

  virtual void addDefinedAtom(Elf_Sym &sym, const DefinedAtom *da,
                              int64_t addr);

  virtual void addUndefinedAtom(Elf_Sym &sym, const UndefinedAtom *ua);

  virtual void addSharedLibAtom(Elf_Sym &sym, const SharedLibraryAtom *sla);

  virtual void finalize(bool sort);

  void write(ELFWriter *writer, TargetLayout<ELFT> &layout,
             llvm::FileOutputBuffer &buffer) override;

  void setStringSection(StringTable<ELFT> *s) { _stringSection = s; }

  StringTable<ELFT> *getStringTable() const { return _stringSection; }

protected:
  struct SymbolEntry {
    SymbolEntry(const Atom *a, const Elf_Sym &sym, const AtomLayout *layout)
        : _atom(a), _atomLayout(layout), _symbol(sym) {}

    const Atom *_atom;
    const AtomLayout *_atomLayout;
    Elf_Sym _symbol;
  };

  llvm::BumpPtrAllocator _symbolAllocate;
  StringTable<ELFT> *_stringSection;
  std::vector<SymbolEntry> _symbolTable;
};

template <class ELFT> class HashSection;

template <class ELFT> class DynamicSymbolTable : public SymbolTable<ELFT> {
public:
  DynamicSymbolTable(const ELFLinkingContext &ctx, TargetLayout<ELFT> &layout,
                     const char *str, int32_t order);

  // Set the dynamic hash table for symbols to be added into
  void setHashTable(HashSection<ELFT> *hashTable) { _hashTable = hashTable; }

  // Add all the dynamic symbos to the hash table
  void addSymbolsToHashTable();

  void finalize() override;

protected:
  HashSection<ELFT> *_hashTable = nullptr;
  TargetLayout<ELFT> &_layout;
};

template <class ELFT> class RelocationTable : public Section<ELFT> {
public:
  typedef llvm::object::Elf_Rel_Impl<ELFT, false> Elf_Rel;
  typedef llvm::object::Elf_Rel_Impl<ELFT, true> Elf_Rela;

  RelocationTable(const ELFLinkingContext &ctx, StringRef str, int32_t order);

  /// \returns the index of the relocation added.
  uint32_t addRelocation(const DefinedAtom &da, const Reference &r);

  bool getRelocationIndex(const Reference &r, uint32_t &res);

  void setSymbolTable(const DynamicSymbolTable<ELFT> *symbolTable) {
    _symbolTable = symbolTable;
  }

  /// \brief Check if any relocation modifies a read-only section.
  bool canModifyReadonlySection() const;

  void finalize() override;

  void write(ELFWriter *writer, TargetLayout<ELFT> &layout,
             llvm::FileOutputBuffer &buffer) override;

protected:
  const DynamicSymbolTable<ELFT> *_symbolTable = nullptr;

  virtual void writeRela(ELFWriter *writer, Elf_Rela &r,
                         const DefinedAtom &atom, const Reference &ref);
  virtual void writeRel(ELFWriter *writer, Elf_Rel &r, const DefinedAtom &atom,
                        const Reference &ref);
  uint32_t getSymbolIndex(const Atom *a);

private:
  std::vector<std::pair<const DefinedAtom *, const Reference *>> _relocs;
};

template <class ELFT> class HashSection;

template <class ELFT> class DynamicTable : public Section<ELFT> {
public:
  typedef llvm::object::Elf_Dyn_Impl<ELFT> Elf_Dyn;
  typedef std::vector<Elf_Dyn> EntriesT;

  DynamicTable(const ELFLinkingContext &ctx, TargetLayout<ELFT> &layout,
               StringRef str, int32_t order);

  range<typename EntriesT::iterator> entries() { return _entries; }

  /// \returns the index of the entry.
  std::size_t addEntry(int64_t tag, uint64_t val);

  void write(ELFWriter *writer, TargetLayout<ELFT> &layout,
             llvm::FileOutputBuffer &buffer) override;

  virtual void createDefaultEntries();
  void doPreFlight() override;

  /// \brief Dynamic table tag for .got.plt section referencing.
  /// Usually but not always targets use DT_PLTGOT for that.
  virtual int64_t getGotPltTag() { return DT_PLTGOT; }

  void finalize() override;

  void setSymbolTable(DynamicSymbolTable<ELFT> *dynsym) {
    _dynamicSymbolTable = dynsym;
  }

  const DynamicSymbolTable<ELFT> *getSymbolTable() const {
    return _dynamicSymbolTable;
  }

  void setHashTable(HashSection<ELFT> *hsh) { _hashTable = hsh; }

  virtual void updateDynamicTable();

protected:
  EntriesT _entries;

  /// \brief Return a virtual address (maybe adjusted) for the atom layout
  /// Some targets like microMIPS and ARM Thumb use the last bit
  /// of a symbol's value to mark 'compressed' code. This function allows
  /// to adjust a virtal address before using it in the dynamic table tag.
  virtual uint64_t getAtomVirtualAddress(const AtomLayout *al) const {
    return al->_virtualAddr;
  }

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
  std::size_t _dt_init_array;
  std::size_t _dt_init_arraysz;
  std::size_t _dt_fini_array;
  std::size_t _dt_fini_arraysz;
  std::size_t _dt_textrel;
  std::size_t _dt_init;
  std::size_t _dt_fini;
  TargetLayout<ELFT> &_layout;
  DynamicSymbolTable<ELFT> *_dynamicSymbolTable;
  HashSection<ELFT> *_hashTable;

  const AtomLayout *getInitAtomLayout();

  const AtomLayout *getFiniAtomLayout();
};

template <class ELFT> class InterpSection : public Section<ELFT> {
public:
  InterpSection(const ELFLinkingContext &ctx, StringRef str, int32_t order,
                StringRef interp);

  void write(ELFWriter *writer, TargetLayout<ELFT> &layout,
             llvm::FileOutputBuffer &buffer);

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
  HashSection(const ELFLinkingContext &ctx, StringRef name, int32_t order);

  /// \brief add the dynamic symbol into the table so that the
  /// hash could be calculated
  void addSymbol(StringRef name, uint32_t index);

  /// \brief Set the dynamic symbol table
  void setSymbolTable(const DynamicSymbolTable<ELFT> *symbolTable);

  // The size of the section has to be determined so that fileoffsets
  // may be properly assigned. Let's calculate the buckets and the chains
  // and fill the chains and the buckets hash table used by the dynamic
  // linker and update the filesize and memory size accordingly
  void doPreFlight() override;

  void finalize() override;

  void write(ELFWriter *writer, TargetLayout<ELFT> &layout,
             llvm::FileOutputBuffer &buffer) override;

private:
  typedef
      typename llvm::object::ELFDataTypeTypedefHelper<ELFT>::Elf_Word Elf_Word;

  std::vector<SymbolTableEntry> _entries;
  std::vector<Elf_Word> _buckets;
  std::vector<Elf_Word> _chains;
  const DynamicSymbolTable<ELFT> *_symbolTable = nullptr;
};

template <class ELFT> class EHFrameHeader : public Section<ELFT> {
public:
  EHFrameHeader(const ELFLinkingContext &ctx, StringRef name,
                TargetLayout<ELFT> &layout, int32_t order);
  void doPreFlight() override;
  void finalize() override;
  void write(ELFWriter *writer, TargetLayout<ELFT> &layout,
             llvm::FileOutputBuffer &buffer) override;

private:
  int32_t _ehFrameOffset = 0;
  TargetLayout<ELFT> &_layout;
};

} // end namespace elf
} // end namespace lld

#endif
