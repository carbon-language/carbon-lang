//===- lib/ReaderWriter/ELF/WriterELF.cpp ---------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/WriterELF.h"
#include "ReferenceKinds.h"

#include "lld/Core/DefinedAtom.h"
#include "lld/Core/File.h"
#include "lld/Core/InputFiles.h"
#include "lld/Core/Reference.h"
#include "lld/Core/SharedLibraryAtom.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include "AtomsELF.h"

#include <map>
#include <unordered_map>
#include <tuple>
#include <vector>

using namespace llvm;
using namespace llvm::object;
namespace lld {
namespace elf {

template<support::endianness target_endianness, bool is64Bits>
class ELFExecutableWriter;

/// \brief The ELFWriter class is a base class for the linker to write
///        various kinds of ELF files.
class ELFWriter : public Writer {
  public:
    ELFWriter() { }

  public:
    /// \brief builds the chunks that needs to be written to the output
    ///        ELF file
    virtual void buildChunks(const lld::File &file) = 0;

    /// \brief Writes the chunks into the output file specified by path
    virtual error_code writeFile(const lld::File &File, StringRef path) = 0;

    /// \brief Writes the chunks into the output file specified by path
    virtual uint64_t addressOfAtom(const Atom *atom) = 0;

    /// \brief Return the processing function to apply Relocations
    virtual KindHandler *kindHandler()  = 0;
};

/// \brief A chunk is a contiguous region of space
template<support::endianness target_endianness, bool is64Bits>
class Chunk {
public:

  /// \brief Describes the type of Chunk
  enum Kind {
    K_ELFHeader, // ELF Header
    K_ELFProgramHeader, // Program Header
    K_ELFSegment, // Segment
    K_ELFSection, // Section
    K_ELFSectionHeader // Section header
  };
  Chunk(StringRef name, Kind kind) 
    : _name(name)
    , _kind(kind)
    , _fsize(0)
    , _msize(0)
    , _align2(0)
    , _order(0)
    , _ordinal(1)
    , _start(0)
    , _fileoffset(0) {}
  virtual             ~Chunk() {}
  // Does the chunk occupy disk space 
  virtual bool        occupiesNoDiskSpace() const {
    return false;    
  }
  // The name of the chunk
  StringRef name() const { return _name; }
  // Kind of chunk
  Kind kind() const { return _kind; }
  uint64_t            fileSize() const { return _fsize; }
  uint64_t            align2() const { return _align2; }
  void                appendAtom() const;

  // The ordinal value of the chunk
  uint64_t            ordinal() const { return _ordinal;}
  void               setOrdinal(uint64_t newVal) { _ordinal = newVal;}
  // The order in which the chunk would appear in the output file
  uint64_t            order() const { return _order; }
  void               setOrder(uint32_t order) { _order = order; }
  // Output file offset of the chunk
  uint64_t            fileOffset() const { return _fileoffset; }
  void               setFileOffset(uint64_t offset) { _fileoffset = offset; }
  // Output start address of the chunk
  void               setVAddr(uint64_t start) { _start = start; }
  uint64_t            virtualAddr() const { return _start; }
  // Does the chunk occupy memory during execution ?
  uint64_t            memSize() const { return _msize; }
  void               setMemSize(uint64_t msize) { _msize = msize; }
  // Writer the chunk 
  virtual void       write(ELFWriter *writer, 
                           OwningPtr<FileOutputBuffer> &buffer) = 0;
  // Finalize the chunk before writing
  virtual void       finalize() = 0;

protected:
  StringRef _name;
  Kind _kind;
  uint64_t _fsize;
  uint64_t _msize;
  uint64_t _align2;
  uint32_t  _order;
  uint64_t _ordinal;
  uint64_t _start;
  uint64_t _fileoffset;
};

/// \brief The ELFLayoutOptions encapsulates the options used by all Layouts
///        Examples of the ELFLayoutOptions would be a script that would be used
///        to drive the layout 
class ELFLayoutOptions {
public:
  ELFLayoutOptions() { }

  ELFLayoutOptions(StringRef &linker_script) : _script(linker_script)
  {}

  /// parse the linker script
  error_code parseLinkerScript();

  /// Is the current section present in the linker script
  bool isSectionPresent();

private:
  StringRef _script;
};

/// \brief The ELFLayout is an abstract class for managing the final layout for 
///        the kind of binaries(Shared Libraries / Relocatables / Executables 0
///        Each architecture (Hexagon, PowerPC, MIPS) would have a concrete 
///        subclass derived from ELFLayout for generating each binary thats 
//         needed by the lld linker
class ELFLayout {
public:
  typedef uint32_t SectionOrder;
  typedef uint32_t SegmentType;
  typedef uint32_t Flags;

public:
  /// Return the order the section would appear in the output file
  virtual SectionOrder getSectionOrder
                        (const StringRef name,
                         int32_t contentType,
                         int32_t contentPerm) = 0;
  /// append the Atom to the layout and create appropriate sections 
  virtual error_code addAtom(const Atom *atom) = 0;
  /// find the Atom Address in the current layout
  virtual bool findAtomAddrByName(const StringRef name, uint64_t &addr) = 0;
  /// associates a section to a segment
  virtual void assignSectionsToSegments() = 0;
  /// associates a virtual address to the segment, section, and the atom
  virtual void assignVirtualAddress() = 0;
  /// associates a file offset to the segment, section and the atom
  virtual void assignFileOffsets() = 0;

public:
  ELFLayout() {}
  ELFLayout(WriterOptionsELF &writerOptions, 
            ELFLayoutOptions &layoutOptions) 
    : _writerOptions(writerOptions)
    , _layoutOptions(layoutOptions) {}
  virtual ~ELFLayout() { }

private:
  WriterOptionsELF _writerOptions;
  ELFLayoutOptions _layoutOptions;
};


/// \brief A section contains a set of atoms that have similiar properties
///        The atoms that have similiar properties are merged to form a section
template<support::endianness target_endianness, bool is64Bits>
class Section : public Chunk<target_endianness, is64Bits> {
public:
  // The Kind of section that the object represents 
  enum SectionKind {
    K_Default,
    K_SymbolTable,
    K_StringTable,
  };
  // Create a section object, the section is set to the default type if the
  // caller doesnot set it
  Section(const StringRef sectionName, 
          const int32_t contentType,
          const int32_t contentPermissions,
          const int32_t order,
          const SectionKind kind = K_Default) 
    : Chunk<target_endianness, is64Bits>
      (sectionName, Chunk<target_endianness, is64Bits>::K_ELFSection)
    , _contentType(contentType)
    , _contentPermissions(contentPermissions)
    , _sectionKind(kind)
    , _entSize(0)
    , _shInfo(0)
    , _link(0) {
    this->setOrder(order);
  }

  /// return the section kind
  SectionKind sectionKind() const {
    return _sectionKind;
  }

  /// Align the offset to the required modulus defined by the atom alignment
  uint64_t alignOffset(uint64_t offset, DefinedAtom::Alignment &atomAlign) {
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
  void appendAtom(const Atom *atom) {
    Atom::Definition atomType = atom->definition();
    const DefinedAtom *definedAtom = dyn_cast<DefinedAtom>(atom);
    assert(atom != nullptr && "Expecting the atom to be a DefinedAtom");
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
        _atoms.push_back(std::make_pair(atom, std::make_pair(fOffset, 0)));
        this->_fsize = fOffset + definedAtom->size();
        this->_msize = mOffset + definedAtom->size();
        break;
      case  DefinedAtom::typeZeroFill:
        _atoms.push_back(std::make_pair(atom, std::make_pair(mOffset, 0)));
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

  /// \brief Set the virtual address of each Atom in the Section. This 
  /// routine gets called after the linker fixes up the virtual address 
  /// of the section
  void assignVirtualAddress(uint64_t &addr) {
    for (auto &ai : _atoms) {
      ai.second.second = addr + ai.second.first;
    }
    addr += this->memSize();
  }

  /// \brief Set the file offset of each Atom in the section. This routine 
  /// gets called after the linker fixes up the section offset 
  void assignOffsets(uint64_t offset) {
    for (auto &ai : _atoms) {
      ai.second.first = offset + ai.second.first;
    }
  }

  /// \brief Find the Atom address given a name, this is needed to to properly 
  ///  apply relocation. The section class calls this to find the atom address
  ///  to fix the relocation
  bool findAtomAddrByName(const StringRef name, uint64_t &addr) {
    for (auto ai : _atoms) {
      if (ai.first->name() == name) {
        addr = ai.second.second;
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

  /// \brief Get the section flags, defined by the permissions of the section
  int64_t flags() {
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

  /// \brief Return the raw flags, we need this to sort segments
  int64_t atomflags() const {
    return _contentPermissions;
  }

  /// \brief Return the section type, the returned value is recorded in the 
  /// sh_type field of the Section Header
  int type() {
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

  /// \brief Returns the section link field, the returned value is 
  ///        recorded in the sh_link field of the Section Header
  int link() const {
    return _link;
  }

  void setLink(int32_t link) {
    _link = link;
  }

  /// \brief Returns the section entsize field, the returned value is 
  ///        recorded in the sh_entsize field of the Section Header
  int entsize() const {
    return _entSize;
  }

  /// \brief Returns the shinfo field, the returned value is 
  ///        recorded in the sh_info field of the Section Header
  int shinfo() const { 
    return _shInfo;
  }

  /// \brief Records the segmentType, that this section belongs to
  void setSegment(const ELFLayout::SegmentType segmentType) {
    _segmentType = segmentType;
  }

  /// \brief convert the segment type to a String for diagnostics
  ///        and printing purposes
  StringRef segmentKindToStr() const {
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

  /// \brief for LLVM style RTTI information
  static inline bool classof(const Chunk<target_endianness, is64Bits> *c) {
    return c->kind() == Chunk<target_endianness, is64Bits>::K_ELFSection;
  }

  /// \brief Finalize the section contents before writing
  void finalize() { }

  /// \brief Write the section and the atom contents to the buffer 
  void write(ELFWriter *writer, 
             OwningPtr<FileOutputBuffer> &buffer) {
    uint8_t *chunkBuffer = buffer->getBufferStart();
    for (auto &ai : _atoms) {
      const DefinedAtom *definedAtom = llvm::dyn_cast<DefinedAtom>(ai.first);
      // Copy raw content of atom to file buffer.
      ArrayRef<uint8_t> content = definedAtom->rawContent();
      uint64_t contentSize = content.size();
      if (contentSize == 0)
        continue;
      uint8_t *atomContent = chunkBuffer + ai.second.first;
      std::copy_n(content.data(), contentSize, atomContent);
      for (auto ref = definedAtom->begin(); ref != definedAtom->end(); ++ref) {
        uint32_t offset = ref->offsetInAtom();
        uint64_t targetAddress = 0;
        if (ref->target() != nullptr)
          targetAddress = writer->addressOfAtom(ref->target());
        else
          assert(0 && "Found the target to be NULL");
        uint64_t fixupAddress = writer->addressOfAtom(ai.first) + offset;
        // apply the relocation
        writer->kindHandler()->applyFixup(ref->kind(), 
                                          ref->addend(),
                                          &atomContent[offset],
                                          fixupAddress,
                                          targetAddress);
      }
    }
  }

  /// Atom Iterators
  typedef typename std::vector<std::pair<const Atom *, 
          std::pair<uint64_t, uint64_t>>>::iterator atom_iter;

  atom_iter atoms_begin() { return _atoms.begin(); }

  atom_iter atoms_end() { return _atoms.end(); }

protected:
  int32_t _contentType;
  int32_t _contentPermissions;
  SectionKind    _sectionKind;
  // An Atom is appended to the vector with the following fields
  // field1 : Atom
  // field2 : fileoffset (initially set with a base offset of 0)
  // field3 : virtual address
  std::vector<std::pair<const Atom *, std::pair<uint64_t, uint64_t>>> _atoms;
  ELFLayout::SegmentType _segmentType;
  int64_t _entSize;
  int64_t _shInfo;
  int64_t _link;
};  

/// \brief A MergedSections represents a set of sections grouped by the same name
///        The output file that gets written by the linker has sections grouped
///        by similiar names
template<support::endianness target_endianness, bool is64Bits>
class MergedSections {
public:
  MergedSections(StringRef name) 
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
                                  
  // Set the MergedSections is associated with a segment
  void setHasSegment() { _hasSegment = true; }

  /// Sets the ordinal 
  void setOrdinal(uint64_t ordinal) {
    _ordinal = ordinal;
  }

  /// Sets the Memory size 
  void setMemSize(uint64_t memsz) { 
    _memSize = memsz;
  }

  /// Sets the size fo the merged Section
  void setSize(uint64_t fsiz) {
    _size = fsiz;
  }

  // The offset of the first section contained in the merged section is contained here
  void setFileOffset(uint64_t foffset) {
    _fileOffset = foffset;
  }

  // Sets the starting address of the section
  void setAddr(uint64_t addr) {
    _virtualAddr = addr;
  }

  // Appends a section into the list of sections that are part of this Merged
  // Section
  void appendSection(Chunk<target_endianness, is64Bits> *c) {
    if (c->align2() > _align2) 
      _align2 = c->align2();
    if (c->kind() == Chunk<target_endianness, is64Bits>::K_ELFSection) {
      Section<target_endianness, is64Bits> *section;
      section = llvm::dyn_cast<Section<target_endianness, is64Bits>>(c);
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

  // Iterators
  typedef typename std::vector<Chunk<target_endianness, is64Bits> *>::iterator 
                                                                     ChunkIter;

  ChunkIter begin_sections() { return _sections.begin(); }

  ChunkIter end_sections() { return _sections.end(); }

  // The below functions returns the properties of the MergeSection
  bool hasSegment() const { return _hasSegment; }

  StringRef name() const { return _name; }

  int64_t shinfo() const { return _shInfo; }

  uint64_t align2() const { return _align2; }

  int64_t link() const { return _link; }

  int64_t type() const { return _type; }

  uint64_t virtualAddr() const { return _virtualAddr; }

  int64_t ordinal() const { return _ordinal; }

  int64_t kind() const { return _kind; }

  uint64_t fileSize() const { return _size; }

  int64_t entsize() const { return _entSize; }

  uint64_t fileOffset() const { return _fileOffset; }

  int64_t flags() const { return _flags; }

  uint64_t memSize() { return _memSize; }

private:
  StringRef _name;
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
  std::vector<Chunk<target_endianness, is64Bits> *> _sections;
};

/// \brief A segment can be divided into segment slices 
///        depending on how the segments can be split 
template<support::endianness target_endianness, bool is64Bits>
class SegmentSlice {
public:
  typedef typename std::vector<Section<target_endianness, is64Bits> *>::iterator 
                                                                    sectionIter;

  SegmentSlice() { }

  /// Set the segment slice so that it begins at the offset specified
  /// by fileoffset and set the start of the slice to be s and the end 
  /// of the slice to be e
  void set(uint64_t fileoffset, int32_t s, int e) {
    _startSection = s;
    _endSection = e+1;
    _offset = fileoffset;
  }

  // Set the segment slice start and end iterators. This is used to walk through
  // the sections that are part of the Segment slice
  void setSections(sectionIter start, sectionIter end) {
    _startSectionIter = start;
    _endSectionIter = end;
  }

  // Return the fileOffset of the slice
  uint64_t fileOffset() const { return _offset; }

  // Return the size of the slice
  uint64_t fileSize() const { return _size; }

  // Return the start of the slice
  int32_t startSection() const { return _startSection; }

  // Return the start address of the slice
  uint64_t virtualAddr() const { return _addr; }

  // Return the memory size of the slice
  uint64_t memSize() const { return _memSize; }

  // Return the alignment of the slice
  uint64_t align2() const { return _align2; }

  void setSize(uint64_t sz) { _size = sz; }

  void setMemSize(uint64_t memsz) { _memSize = memsz; }

  void setVAddr(uint64_t addr) { _addr = addr; }

  void setAlign(uint64_t align) { _align2 = align; }

  static bool compare_slices(SegmentSlice<target_endianness, is64Bits> *a, 
                             SegmentSlice<target_endianness, is64Bits> *b) {
    return (a->startSection() < b->startSection());
  }

  // Functions to run through the slice
  sectionIter sections_begin() { return _startSectionIter; }

  sectionIter sections_end() { return _endSectionIter; }

private:
  int32_t _startSection;
  int32_t _endSection;
  sectionIter _startSectionIter;
  sectionIter _endSectionIter;
  uint64_t _addr;
  uint64_t _offset;
  uint64_t _size;
  uint64_t _align2;
  uint64_t _memSize;
};

/// \brief A segment contains a set of sections, that have similiar properties
//  the sections are already seperated based on different flags and properties
//  the segment is just a way to concatenate sections to segments
template<support::endianness target_endianness, bool is64Bits>
class Segment : public Chunk<target_endianness, is64Bits> {
public:
  typedef typename std::vector
       <SegmentSlice<target_endianness, is64Bits> *>::iterator slice_iter;
  typedef typename 
        std::vector<Section<target_endianness, is64Bits> *>::iterator SectionIter;

  Segment(const StringRef name,
          const ELFLayout::SegmentType type,
          const WriterOptionsELF &options)
    : Chunk<target_endianness, is64Bits>(name
    , Chunk<target_endianness, is64Bits>::K_ELFSegment)
    , _segmentType(type)
    , _flags(0)
    , _atomflags(0)
    , _options(options) {
    this->_align2 = 0;
    this->_fsize = 0;
  }

  /// append a section to a segment
  void append(Section<target_endianness, is64Bits> *section) {
    _sections.push_back(section);
    if (_flags < section->flags())
      _flags = section->flags();
    if (_atomflags < section->atomflags())
      _atomflags = section->atomflags();
    if (this->_align2 < section->align2())
      this->_align2 = section->align2();
  }

  /// Sort segments depending on the property
  /// If we have a Program Header segment, it should appear first
  /// If we have a INTERP segment, that should appear after the Program Header
  /// All Loadable segments appear next in this order
  /// All Read Write Execute segments follow
  /// All Read Execute segments appear next
  /// All Read only segments appear first
  /// All Write execute segments follow 
  static bool compareSegments(Segment<target_endianness, is64Bits> *sega, 
                              Segment<target_endianness, is64Bits> *segb)
  {

    if (sega->atomflags() < segb->atomflags()) 
      return false;
    return true;
  }

  /// \brief Start assigning file offset to the segment chunks The fileoffset
  /// needs to be page at the start of the segment and in addition the
  /// fileoffset needs to be aligned to the max section alignment within the
  /// segment. This is required so that the ELF property p_poffset % p_align =
  /// p_vaddr mod p_align holds true.
  /// The algorithm starts off by assigning the startOffset thats passed in as
  /// parameter to the first section in the segment, if the difference between
  /// the newly computed offset is greater than a page, then we create a segment
  /// slice, as it would be a waste of virtual memory just to be filled with 
  /// zeroes
  void assignOffsets(uint64_t startOffset) {
    int startSection = 0;
    int currSection = 0;
    SectionIter startSectionIter, endSectionIter;
    // slice align is set to the max alignment of the chunks that are 
    // contained in the slice
    uint64_t sliceAlign = 0;
    // Current slice size
    uint64_t curSliceSize = 0;
    // Current Slice File Offset
    uint64_t curSliceFileOffset = 0;

    startSectionIter = _sections.begin();
    endSectionIter = _sections.end();
    startSection = 0;
    bool isFirstSection = true;
    for (auto si = _sections.begin(); si != _sections.end(); ++si) {
      if (isFirstSection) {
        // align the startOffset to the section alignment
        uint64_t newOffset = llvm::RoundUpToAlignment(startOffset, (*si)->align2());
        curSliceFileOffset = newOffset;
        sliceAlign = (*si)->align2();
        this->setFileOffset(startOffset);
        (*si)->setFileOffset(newOffset);
        curSliceSize = (*si)->fileSize();
        isFirstSection = false;
      } else {
        uint64_t curOffset = curSliceFileOffset + curSliceSize;
        uint64_t newOffset = llvm::RoundUpToAlignment(curOffset, (*si)->align2());
        SegmentSlice<target_endianness, is64Bits> *slice = nullptr;
        // If the newOffset computed is more than a page away, lets create
        // a seperate segment, so that memory is not used up while running
        if ((newOffset - curOffset) > _options.pageSize()) {
          // TODO: use std::find here
          for (auto sei = slices_begin(); sei != slices_end(); ++sei) {
            if ((*sei)->startSection() == startSection) {
              slice = *sei;
              break;
            }
          }
          if (!slice) {
            slice = new (_segmentAllocate.Allocate
                         <SegmentSlice<target_endianness, is64Bits>>()) 
                         SegmentSlice<target_endianness, is64Bits>();
            _segmentSlices.push_back(slice);
          }
          slice->set(curSliceFileOffset, startSection, currSection);
          slice->setSections(startSectionIter, endSectionIter);
          slice->setSize(curSliceSize);
          slice->setAlign(sliceAlign);
          uint64_t newPageOffset = llvm::RoundUpToAlignment(curOffset, _options.pageSize());
          newOffset = llvm::RoundUpToAlignment(newPageOffset, (*si)->align2());
          curSliceFileOffset = newOffset;
          startSectionIter = endSectionIter;
          startSection = currSection;
          (*si)->setFileOffset(curSliceFileOffset);
          curSliceSize = newOffset - curSliceFileOffset + (*si)->fileSize();
          sliceAlign = (*si)->align2();
        }
        else {
          if (sliceAlign < (*si)->align2())
            sliceAlign = (*si)->align2();
          (*si)->setFileOffset(newOffset);
          curSliceSize = newOffset - curSliceFileOffset + (*si)->fileSize();
        }
      }
      currSection++;
      endSectionIter = si;
    }
    SegmentSlice<target_endianness, is64Bits> *slice = nullptr;
    for (auto sei = slices_begin(); sei != slices_end(); ++sei) {
      // TODO: add std::find
      if ((*sei)->startSection() == startSection) {
        slice = *sei;
        break;
      }
    }
    if (!slice) {
      slice = new (_segmentAllocate.Allocate
                   <SegmentSlice<target_endianness, is64Bits>>()) 
                   SegmentSlice<target_endianness, is64Bits>();
      _segmentSlices.push_back(slice);
    }
    slice->set(curSliceFileOffset, startSection, currSection);
    slice->setSections(startSectionIter, _sections.end());
    slice->setSize(curSliceSize);
    slice->setAlign(sliceAlign);
    this->_fsize = curSliceFileOffset - startOffset + curSliceSize;
    std::stable_sort(slices_begin(), slices_end(), 
                     SegmentSlice<target_endianness, is64Bits>::compare_slices);
  }

  /// \brief Assign virtual addresses to the slices
  void assignVirtualAddress(uint64_t &addr, bool isFirstSegment) {
    for (auto sei = slices_begin(), see = slices_end(); sei != see; ++sei) {
      bool firstSlice = (sei == slices_begin());
      // The first segment has distinct since it contains the 
      // ELF header and the Program Header, if we get to the first segment
      // and the first slice, set it to the baseaddress
      // which is the segment address
      if (isFirstSegment && firstSlice)
        (*sei)->setVAddr(this->virtualAddr());
      else {
        // Align to a page
        addr = llvm::RoundUpToAlignment(addr, _options.pageSize());
        // Align to the slice alignment
        addr = llvm::RoundUpToAlignment(addr, (*sei)->align2());
      }
      bool virtualAddressSet = false;
      for (auto si = (*sei)->sections_begin(), se = (*sei)->sections_end(); 
                                               si != se; ++si) {
        // Align the section address
        addr = llvm::RoundUpToAlignment(addr, (*si)->align2());
        if (!isFirstSegment && !virtualAddressSet) {
          (*sei)->setVAddr(addr);
          virtualAddressSet = true;
        }
        (*si)->setVAddr(addr);
        (*si)->assignVirtualAddress(addr);
        (*si)->setMemSize(addr - (*si)->virtualAddr());
      }
      (*sei)->setMemSize(addr - (*sei)->virtualAddr());
    }
  }

  slice_iter slices_begin() {
    return _segmentSlices.begin();
  }

  slice_iter slices_end() {
    return _segmentSlices.end();
  }

  // Write the Segment 
  void write(ELFWriter *writer, OwningPtr<FileOutputBuffer> &buffer) {
    for (auto sei = slices_begin(), see = slices_end(); sei != see; ++sei) {
      for (auto si = (*sei)->sections_begin(), se = (*sei)->sections_end(); 
                                               si != se; ++si) {
        (*si)->write(writer, buffer);
      }
    }
  }

  // Finalize the segment, before we want to write to the output file
  void finalize() { }

  // For LLVM RTTI 
  static inline bool classof(const Chunk<target_endianness, is64Bits> *c) {
    return c->kind() == Chunk<target_endianness, is64Bits>::K_ELFSegment;
  }

  // Getters
  int32_t sectionCount() const {
    return _sections.size();
  }

  ELFLayout::SegmentType segmentType() { return _segmentType; }

  int pageSize() const { return _options.pageSize(); }

  int64_t atomflags() const { return _atomflags; }

  int64_t flags() const {
    int64_t fl = 0;
    if (_flags & llvm::ELF::SHF_ALLOC)
      fl |= llvm::ELF::PF_R;
    if (_flags & llvm::ELF::SHF_WRITE)
      fl |= llvm::ELF::PF_W;
    if (_flags & llvm::ELF::SHF_EXECINSTR)
      fl |= llvm::ELF::PF_X;
    return fl;
  }

  int64_t numSlices() const {
    return _segmentSlices.size();
  }

private:
  std::vector<Section<target_endianness, is64Bits> *> _sections;
  std::vector<SegmentSlice<target_endianness, is64Bits> *> _segmentSlices;
  ELFLayout::SegmentType _segmentType;
  int64_t _flags;
  int64_t _atomflags;
  const WriterOptionsELF _options;
  llvm::BumpPtrAllocator _segmentAllocate;
};

/// \brief The class represents the ELF String Table
template<support::endianness target_endianness, bool is64Bits>
class ELFStringTable : public Section<target_endianness, is64Bits> {

public:
  ELFStringTable(const char *str, 
                 int32_t order): 
                            Section<target_endianness, is64Bits>(str,
                                                         llvm::ELF::SHT_STRTAB,
                                                         DefinedAtom::perm___,
                                                         order,
                            Section<target_endianness, is64Bits>::K_StringTable) {
    // the string table has a NULL entry for which
    // add an empty string
    _strings.push_back("");
    this->_fsize = 1;
    this->_align2 = 1;
    this->setOrder(order);
  }

  static inline bool classof(ELFStringTable<target_endianness, is64Bits> *s) {
    return true;
  }

  static inline bool classof(const Chunk<target_endianness, is64Bits> *c) {
    return c->kind() == Section<target_endianness, is64Bits>::K_StringTable;
  }

  uint64_t addString(const StringRef symname) {
    _strings.push_back(symname);
    uint64_t offset = this->_fsize;
    this->_fsize += symname.size() + 1;
    return offset;
  }

  void write(ELFWriter *writer, 
             OwningPtr<FileOutputBuffer> &buffer) {
    uint8_t *chunkBuffer = buffer->getBufferStart();
    uint8_t *dest = chunkBuffer + this->fileOffset();
    for (auto si : _strings) {
      memcpy(dest, si.data(), si.size());
      dest += si.size();
      memcpy(dest, "", 1);
      dest += 1;
    }
  }

  void finalize() { }

private:
  std::vector<StringRef> _strings;
};

/// \brief The ELFSymbolTable class represents the symbol table in a ELF file 
template<support::endianness target_endianness, bool is64Bits>
class ELFSymbolTable : public Section<target_endianness, is64Bits> {

public:
  typedef object::Elf_Sym_Impl<target_endianness, is64Bits> Elf_Sym;

  ELFSymbolTable(const char *str,
                 int32_t order)
    : Section<target_endianness, is64Bits>(str
    , llvm::ELF::SHT_SYMTAB
    , 0
    , order
    , Section<target_endianness, is64Bits>::K_SymbolTable) {
    this->setOrder(order);
    Elf_Sym *symbol = new (_symbolAllocate.Allocate<Elf_Sym>()) Elf_Sym;
    memset ((void *)symbol,0, sizeof(Elf_Sym));
    _symbolTable.push_back(symbol);
    this->_entSize = sizeof(Elf_Sym);
    this->_fsize = sizeof(Elf_Sym);
    this->_align2 = sizeof(void *);
  }

  static inline bool classof(ELFSymbolTable<target_endianness, is64Bits> *s) {
    return true;
  }

  static inline bool classof(const Chunk<target_endianness, is64Bits> *c) {
    return c->kind() == Section<target_endianness, is64Bits>::K_SymbolTable;
  }

  void addSymbol(const Atom *atom, int32_t sectionIndex, uint64_t addr = 0) {
    Elf_Sym *symbol = new(_symbolAllocate.Allocate<Elf_Sym>()) Elf_Sym;
    unsigned char binding = 0, type = 0;
    symbol->st_name = _stringSection->addString(atom->name());
    symbol->st_size = 0;
    symbol->st_shndx = sectionIndex;
    symbol->st_value = 0;
    symbol->st_other = ELF::STV_DEFAULT;
    if (const DefinedAtom *da = llvm::dyn_cast<const DefinedAtom>(atom)){
      symbol->st_size = da->size();
      lld::DefinedAtom::ContentType ct;
      switch (ct = da->contentType()){
      case  DefinedAtom::typeCode:
        symbol->st_value = addr;
        type = ELF::STT_FUNC;
        break;
      case  DefinedAtom::typeData:
        symbol->st_value = addr;
        type = ELF::STT_OBJECT;
        break;
      case  DefinedAtom::typeZeroFill:
        type = ELF::STT_COMMON;
        symbol->st_value = addr;
        break;
      default:
        type = ELF::STT_NOTYPE;
      }
      if (da->scope() == DefinedAtom::scopeTranslationUnit)  
        binding = ELF::STB_LOCAL;
      else
        binding = ELF::STB_GLOBAL;
    } else if (const AbsoluteAtom *aa = llvm::dyn_cast<const AbsoluteAtom>(atom)){
      type = ELF::STT_OBJECT;
      symbol->st_shndx = ELF::SHN_ABS;
      switch (aa->scope()) {
      case AbsoluteAtom::scopeLinkageUnit:
        symbol->st_other = ELF::STV_HIDDEN;
        binding = ELF::STB_LOCAL;
        break;
      case AbsoluteAtom::scopeTranslationUnit:
        binding = ELF::STB_LOCAL;
        break;
      case AbsoluteAtom::scopeGlobal:
        binding = ELF::STB_GLOBAL;
        break;
      }
      symbol->st_value = aa->value();
    } 
    else {
     symbol->st_value = 0;
     type = ELF::STT_NOTYPE;
     binding = ELF::STB_WEAK;
    }
    symbol->setBindingAndType(binding, type);
    _symbolTable.push_back(symbol);
    this->_fsize += sizeof(Elf_Sym);
  }

  void setStringSection(ELFStringTable<target_endianness, is64Bits> *s) {
    _stringSection = s;
  }

  void finalize() {
    // sh_info should be one greater than last symbol with STB_LOCAL binding
    // we sort the symbol table to keep all local symbols at the beginning
    std::stable_sort(_symbolTable.begin(), _symbolTable.end(),
    [](const Elf_Sym *A, const Elf_Sym *B) {
       return A->getBinding() < B->getBinding();
    }); 
    uint16_t shInfo = 0;
    for (auto i : _symbolTable) {
      if (i->getBinding() != ELF::STB_LOCAL)
        break;
      shInfo++;
    }
    this->_shInfo = shInfo;
    this->setLink(_stringSection->ordinal());
  }

  void write(ELFWriter *writer, 
             OwningPtr<FileOutputBuffer> &buffer) {

    uint8_t *chunkBuffer = buffer->getBufferStart();
    uint8_t *dest = chunkBuffer + this->fileOffset();
    for (auto sti : _symbolTable) {
      memcpy(dest, sti, sizeof(Elf_Sym));
      dest += sizeof(Elf_Sym);
    }
  }

private:
  ELFStringTable<target_endianness, is64Bits> *_stringSection;
  std::vector<Elf_Sym*> _symbolTable;
  llvm::BumpPtrAllocator _symbolAllocate;
  int64_t _link;
};

/// \brief An ELFHeader represents the Elf[32/64]_Ehdr structure at the
///        start of an ELF executable file.
template<support::endianness target_endianness, bool is64Bits>
class ELFHeader : public Chunk<target_endianness, is64Bits> {
public:
  typedef object::Elf_Ehdr_Impl<target_endianness, is64Bits> Elf_Ehdr;

  ELFHeader()
  : Chunk<target_endianness, is64Bits>("elfhdr"
  , Chunk<target_endianness, is64Bits>::K_ELFHeader) {
    memset(_eh.e_ident, 0, llvm::ELF::EI_NIDENT);
    e_ident(ELF::EI_MAG0, 0x7f);
    e_ident(ELF::EI_MAG1, 'E');
    e_ident(ELF::EI_MAG2, 'L');
    e_ident(ELF::EI_MAG3, 'F');
    e_ehsize(sizeof(Elf_Ehdr));
    e_flags(2);
  }
  void e_ident(int I, unsigned char C) { _eh.e_ident[I] = C; }
  void e_type(uint16_t type)           { _eh.e_type = type; }
  void e_machine(uint16_t machine)     { _eh.e_machine = machine; }
  void e_version(uint32_t version)     { _eh.e_version = version; }
  void e_entry(int64_t entry)         { _eh.e_entry = entry; }
  void e_phoff(int64_t phoff)         { _eh.e_phoff = phoff; }
  void e_shoff(int64_t shoff)         { _eh.e_shoff = shoff; }
  void e_flags(uint32_t flags)         { _eh.e_flags = flags; }
  void e_ehsize(uint16_t ehsize)       { _eh.e_ehsize = ehsize; }
  void e_phentsize(uint16_t phentsize) { _eh.e_phentsize = phentsize; }
  void e_phnum(uint16_t phnum)         { _eh.e_phnum = phnum; }
  void e_shentsize(uint16_t shentsize) { _eh.e_shentsize = shentsize; }
  void e_shnum(uint16_t shnum)         { _eh.e_shnum = shnum; }
  void e_shstrndx(uint16_t shstrndx)   { _eh.e_shstrndx = shstrndx; }
  uint64_t  fileSize()                 { return sizeof (Elf_Ehdr); }

  static inline bool classof(const Chunk<target_endianness, is64Bits> *c) {
    return c->Kind() ==  Chunk<target_endianness, is64Bits>::K_ELFHeader;
  }

  void write(ELFWriter *writer, 
             OwningPtr<FileOutputBuffer> &buffer) {

    uint8_t *chunkBuffer = buffer->getBufferStart();
    uint8_t *atomContent = chunkBuffer + this->fileOffset();
    memcpy(atomContent, &_eh, fileSize());
  }

  void finalize() { }

private:
  Elf_Ehdr _eh;
};

/// \brief An ELFProgramHeader represents the Elf[32/64]_Phdr structure at the
///        start of an ELF executable file.
template<support::endianness target_endianness, bool is64Bits>
class ELFProgramHeader : public Chunk<target_endianness, is64Bits> {
public:
  typedef object::Elf_Phdr<target_endianness, is64Bits> Elf_Phdr;

  ELFProgramHeader()
  : Chunk<target_endianness, is64Bits>("elfphdr"
  , Chunk<target_endianness, is64Bits>::K_ELFProgramHeader) { }

  bool addSegment(Segment<target_endianness, is64Bits> *segment) {
    Elf_Phdr *phdr = nullptr;
    bool ret = false;

    for (auto sei = segment->slices_begin(), see = segment->slices_end(); 
                                             sei != see; ++sei) {
      if (_phi == _ph.end()) {
        phdr = new(_allocator.Allocate<Elf_Phdr>()) Elf_Phdr;
        _ph.push_back(phdr);
        _phi = _ph.end();
        ret = true;
      } else {
        phdr = (*_phi);
        ++_phi;
      }
      phdr->p_type = segment->segmentType();
      phdr->p_offset = (*sei)->fileOffset();
      phdr->p_vaddr = (*sei)->virtualAddr();
      phdr->p_paddr = (*sei)->virtualAddr();
      phdr->p_filesz = (*sei)->fileSize();
      phdr->p_memsz = (*sei)->memSize();
      phdr->p_flags = segment->flags();
      phdr->p_align = (phdr->p_type == llvm::ELF::PT_LOAD) ? 
                       segment->pageSize() : (*sei)->align2();
    }
    return ret;
  }

  void resetProgramHeaders() {
    _phi = _ph.begin();    
  }

  void setVAddr(uint64_t addr) {
    this->_start = llvm::RoundUpToAlignment(addr, 8);
    this->_fsize = this->_start - addr;
  }

  uint64_t  fileSize() { return this->_fsize + (sizeof (Elf_Phdr) * _ph.size()); }

  static inline bool classof(const Chunk<target_endianness, is64Bits> *c) {
    return c->Kind() ==  Chunk<target_endianness, is64Bits>::K_ELFProgramHeader;
  }

  void write(ELFWriter *writer, 
             OwningPtr<FileOutputBuffer> &buffer) {

    uint8_t *chunkBuffer = buffer->getBufferStart();
    uint8_t *dest = chunkBuffer + this->fileOffset();
    for (auto phi : _ph) {
      memcpy(dest, phi, sizeof(Elf_Phdr));
      dest += sizeof(Elf_Phdr);
    }
  }

  void finalize() { }

  int64_t entsize() { 
    return sizeof(Elf_Phdr);
  }

  int64_t numHeaders() {
    return _ph.size();
  }

private:
  std::vector<Elf_Phdr *> _ph;
  typedef typename std::vector<Elf_Phdr *>::iterator ph_iter;
  ph_iter _phi;
  llvm::BumpPtrAllocator  _allocator;
};

/// \brief An ELFSectionHeader represents the Elf[32/64]_Shdr structure
/// at the end of the file
template<support::endianness target_endianness, bool is64Bits>
class ELFSectionHeader : public Chunk<target_endianness, is64Bits> {
public:
  typedef object::Elf_Shdr_Impl<target_endianness, is64Bits> Elf_Shdr;

  ELFSectionHeader(int32_t order):Chunk<target_endianness, is64Bits>("shdr",
                                 Chunk<target_endianness, is64Bits>::K_ELFSectionHeader)
  {
    this->_fsize = 0;
    this->_align2 = 8;
    this->setOrder(order);
    // The first element in the list is always NULL
    Elf_Shdr *nullshdr = new (_sectionAllocate.Allocate<Elf_Shdr>()) Elf_Shdr;
    ::memset(nullshdr, 0, sizeof (Elf_Shdr));
    _sectionInfo.push_back(nullshdr);
    this->_fsize += sizeof (Elf_Shdr);
  }

  uint16_t fileSize() { 
    return sizeof(Elf_Shdr) * _sectionInfo.size();
  }

  void appendSection(MergedSections<target_endianness, is64Bits> *section) { 
    Elf_Shdr *shdr = new (_sectionAllocate.Allocate<Elf_Shdr>()) Elf_Shdr;
    shdr->sh_name   = _stringSection->addString(section->name());
    shdr->sh_type   = section->type();
    shdr->sh_flags  = section->flags();
    shdr->sh_offset = section->fileOffset();
    shdr->sh_addr   = section->virtualAddr();
    shdr->sh_size   = section->memSize();
    shdr->sh_link   = section->link();
    shdr->sh_info   = section->shinfo();
    shdr->sh_addralign = section->align2();
    shdr->sh_entsize = section->entsize();
    _sectionInfo.push_back(shdr);
  }

  void updateSection(Section<target_endianness, is64Bits> *section) {
    Elf_Shdr *shdr = _sectionInfo[section->ordinal()];
    shdr->sh_type   = section->type();
    shdr->sh_flags  = section->flags();
    shdr->sh_offset = section->fileOffset();
    shdr->sh_addr   = section->virtualAddr();
    shdr->sh_size   = section->fileSize();
    shdr->sh_link   = section->link();
    shdr->sh_info   = section->shinfo();
    shdr->sh_addralign = section->align2();
    shdr->sh_entsize = section->entsize();
  }

  static inline bool classof(const Chunk<target_endianness, is64Bits> *c) {
    return c->getChunkKind() ==  Chunk<target_endianness, is64Bits>::K_ELFSectionHeader;
  }

  void setStringSection(ELFStringTable<target_endianness, is64Bits> *s) {
    _stringSection = s;
  }

  void write(ELFWriter *writer, 
             OwningPtr<FileOutputBuffer> &buffer) {

    uint8_t *chunkBuffer = buffer->getBufferStart();
    uint8_t *dest = chunkBuffer + this->fileOffset();
    for (auto shi : _sectionInfo) {
      memcpy(dest, shi, sizeof(Elf_Shdr));
      dest += sizeof(Elf_Shdr);
    }
    _stringSection->write(writer, buffer);
  }

  void finalize() { }

  int64_t entsize() { 
    return sizeof(Elf_Shdr);
  }

  int64_t numHeaders() {
    return _sectionInfo.size();
  }

private:
  ELFStringTable<target_endianness, is64Bits> *_stringSection;
  std::vector<Elf_Shdr*>                  _sectionInfo;
  llvm::BumpPtrAllocator                  _sectionAllocate;
};


/// \brief The DefaultELFLayout class is used by the Writer to arrange 
///        sections and segments in the order determined by the target ELF 
///        format. The writer creates a single instance of the DefaultELFLayout
///        class 
template<support::endianness target_endianness, bool is64Bits>
class DefaultELFLayout : public ELFLayout {
public:

  // The order in which the sections appear in the output file
  // If its determined, that the layout needs to change 
  // just changing the order of enumerations would essentially 
  // change the layout in the output file
  enum DefaultSectionOrder {
    ORDER_NOT_DEFINED = 0,
    ORDER_INTERP,
    ORDER_NOTE,
    ORDER_HASH,
    ORDER_DYNAMIC_SYMBOLS,
    ORDER_DYNAMIC_STRINGS,
    ORDER_INIT,
    ORDER_TEXT,
    ORDER_PLT,
    ORDER_FINI,
    ORDER_RODATA,
    ORDER_EH_FRAME,
    ORDER_EH_FRAMEHDR,
    ORDER_CTORS,
    ORDER_DTORS,
    ORDER_DYNAMIC,
    ORDER_GOT,
    ORDER_GOT_PLT,
    ORDER_DATA,
    ORDER_BSS,
    ORDER_OTHER,
    ORDER_SECTION_STRINGS,
    ORDER_SYMBOL_TABLE,
    ORDER_STRING_TABLE,
    ORDER_SECTION_HEADERS
  };

public:

  // The Key used for creating Sections
  // The sections are created using 
  // SectionName, [contentType, contentPermissions]
  typedef std::pair<StringRef, 
                    std::pair<int32_t, int32_t>> Key;
  typedef typename std::vector<Chunk<target_endianness, is64Bits> *>::iterator 
                                                                     ChunkIter;
  // The key used for Segments
  // The segments are created using 
  // SegmentName, Segment flags
  typedef std::pair<StringRef, int64_t> SegmentKey;
  // Merged Sections contain the map of Sectionnames to a vector of sections,
  // that have been merged to form a single section
  typedef std::map<StringRef, MergedSections<target_endianness, is64Bits> *> 
                                                            MergedSectionMapT;
  typedef typename std::vector
   <MergedSections<target_endianness, is64Bits> *>::iterator MergedSectionIter;

  // HashKey for the Section
  class HashKey {
  public:
    int64_t operator() (const Key &k) const {
      // k.first = section Name
      // k.second = [contentType, Permissions]
      return llvm::hash_combine(k.first, k.second.first, k.second.second);
    }
  };

  // HashKey for the Segment
  class SegmentHashKey {
  public:
    int64_t operator() (const SegmentKey &k) const {
      // k.first = SegmentName
      // k.second = SegmentFlags
      return llvm::hash_combine(k.first, k.second);
    }
  };

  typedef std::unordered_map<Key, 
          Section<target_endianness, is64Bits>*, HashKey> SectionMapT;
  typedef std::unordered_map<SegmentKey,
                             Segment<target_endianness, is64Bits>*, 
                             SegmentHashKey> SegmentMapT;

  DefaultELFLayout(const WriterOptionsELF &options):_options(options) { }

  /// \brief Return the section order for a input section
  virtual SectionOrder getSectionOrder
              (const StringRef name,
              int32_t contentType,
              int32_t contentPermissions) {
    switch (contentType) {
    case DefinedAtom::typeCode:
      return llvm::StringSwitch<Reference::Kind>(name)
        .StartsWith(".eh_frame_hdr", ORDER_EH_FRAMEHDR)
        .StartsWith(".eh_frame", ORDER_EH_FRAME)
        .StartsWith(".init", ORDER_INIT)
        .StartsWith(".fini", ORDER_FINI)
        .StartsWith(".hash", ORDER_HASH)
        .Default(ORDER_TEXT);

    case DefinedAtom::typeConstant:
      return ORDER_RODATA;        
  
    case DefinedAtom::typeData:
      return ORDER_DATA;
  
    case DefinedAtom::typeZeroFill:
      return ORDER_BSS;

    default:
      // If we get passed in a section push it to OTHER
      if (contentPermissions == DefinedAtom::perm___) 
        return ORDER_OTHER;
    
      return ORDER_NOT_DEFINED;
    }
  }

  /// \brief This maps the input sections to the output section names
  StringRef getSectionName(const StringRef name,
                           const int32_t contentType) {
    if (contentType == DefinedAtom::typeZeroFill) 
      return ".bss";
    if (name.startswith(".text")) 
      return ".text";
    if (name.startswith(".rodata")) 
      return ".rodata";
    return name;
  }

  /// \brief Gets the segment for a output section
  virtual ELFLayout::SegmentType getSegmentType(Section<target_endianness, 
                                     is64Bits> *section) const {
    switch(section->order()) {
    case ORDER_INTERP:
      return llvm::ELF::PT_INTERP;

    case ORDER_TEXT:
    case ORDER_HASH:
    case ORDER_DYNAMIC_SYMBOLS:
    case ORDER_DYNAMIC_STRINGS:
    case ORDER_INIT:
    case ORDER_PLT:
    case ORDER_FINI:
    case ORDER_RODATA:
    case ORDER_EH_FRAME:
    case ORDER_EH_FRAMEHDR:
      return llvm::ELF::PT_LOAD;

    case ORDER_NOTE:
      return llvm::ELF::PT_NOTE;

    case ORDER_DYNAMIC:
      return llvm::ELF::PT_DYNAMIC;

    case ORDER_CTORS:
    case ORDER_DTORS:
    case ORDER_GOT:
      return llvm::ELF::PT_GNU_RELRO;

    case ORDER_GOT_PLT:
    case ORDER_DATA:
    case ORDER_BSS:
      return llvm::ELF::PT_LOAD;

    default:
      return llvm::ELF::PT_NULL;
    }
  }

  /// \brief Returns true/false depending on whether the section has a Output
  //         segment or not
  static bool hasOutputSegment(Section<target_endianness, 
                                 is64Bits> *section) {
    switch(section->order()) {
    case ORDER_INTERP:
    case ORDER_HASH:
    case ORDER_DYNAMIC_SYMBOLS:
    case ORDER_DYNAMIC_STRINGS:
    case ORDER_INIT:
    case ORDER_PLT:
    case ORDER_TEXT:
    case ORDER_FINI:
    case ORDER_RODATA:
    case ORDER_EH_FRAME:
    case ORDER_EH_FRAMEHDR:
    case ORDER_NOTE:
    case ORDER_DYNAMIC:
    case ORDER_CTORS:
    case ORDER_DTORS:
    case ORDER_GOT:
    case ORDER_GOT_PLT:
    case ORDER_DATA:
    case ORDER_BSS:
      return true;

    default:
      return false;
    }
  }

  // Adds an atom to the section
  virtual error_code addAtom(const Atom *atom) {
    const DefinedAtom *definedAtom = dyn_cast<DefinedAtom>(atom);
    const StringRef sectionName = 
                getSectionName(definedAtom->customSectionName(),
                               definedAtom->contentType());
    const lld::DefinedAtom::ContentPermissions permissions = 
                                  definedAtom->permissions();
    const lld::DefinedAtom::ContentType contentType = 
                                  definedAtom->contentType();
    const Key key(sectionName, std::make_pair(contentType, permissions));
    const std::pair<Key, Section<target_endianness, is64Bits> *> 
                                                currentSection(key, nullptr);
    std::pair<typename SectionMapT::iterator, bool> 
                              sectionInsert(_sectionMap.insert(currentSection));
    Section<target_endianness, is64Bits> *section;
    // the section is already in the map
    if (!sectionInsert.second) {
      section = sectionInsert.first->second;
      section->setContentPermissions(permissions);
    }
    else {
      SectionOrder section_order = getSectionOrder(sectionName,
                                     contentType,
                                     permissions);
      section = new (_allocator.Allocate
                     <Section<target_endianness, is64Bits>>())
                     Section<target_endianness, is64Bits>
                     (sectionName, contentType, 
                      permissions, section_order);
      sectionInsert.first->second = section;
      section->setOrder(section_order);
      _sections.push_back(section);
    }
    section->appendAtom(atom);
    return error_code::success();
  }

  // Merge sections with the same name into a MergedSections
  void mergeSimiliarSections() {
    MergedSections<target_endianness, is64Bits> *mergedSection;
  
    for (auto &si : _sections) {
      const std::pair<StringRef, MergedSections<target_endianness, is64Bits> *> 
                                            currentMergedSections(si->name(), 
                                                                 nullptr);
      std::pair<typename MergedSectionMapT::iterator, bool>
                              mergedSectionInsert
                              (_mergedSectionMap.insert(currentMergedSections));
      if (!mergedSectionInsert.second) {
        mergedSection = mergedSectionInsert.first->second;
      }
      else {
        mergedSection = new (_allocator.Allocate
                     <MergedSections<target_endianness, is64Bits>>())
                     MergedSections<target_endianness, is64Bits>(si->name());
        _mergedSections.push_back(mergedSection);
        mergedSectionInsert.first->second = mergedSection;
      }
      mergedSection->appendSection(si);
    }
  }

  void assignSectionsToSegments() {
    // sort the sections by their order as defined by the layout
    std::stable_sort(_sections.begin(), _sections.end(),
    [](Chunk<target_endianness, is64Bits> *A,
       Chunk<target_endianness, is64Bits> *B) {
       return A->order() < B->order();
    });
    // Merge all sections
    mergeSimiliarSections();
    // Set the ordinal after sorting the sections
    int ordinal = 1;
    for (auto &msi : _mergedSections) {
      (*msi).setOrdinal(ordinal);
      for (auto ai = (*msi).begin_sections(), ae = (*msi).end_sections();
                                               ai != ae; ++ai) {
        (*ai)->setOrdinal(ordinal);
      }
      ++ordinal;
    }
    Section<target_endianness, is64Bits> *section;
    Segment<target_endianness, is64Bits> *segment;
    for (auto msi = merged_sections_begin(), mse = merged_sections_end(); 
                                             msi != mse; ++msi) {
      for (auto ai = (*msi)->begin_sections(), ae = (*msi)->end_sections(); 
                                               ai != ae; ++ai) {
        if ((*ai)->kind() == Chunk<target_endianness, is64Bits>::K_ELFSection) {
          section = llvm::dyn_cast<Section<target_endianness, is64Bits>>(*ai);
          if (!hasOutputSegment(section))
            continue;
          (*msi)->setHasSegment();
          section->setSegment(getSegmentType(section));
          const StringRef segmentName = section->segmentKindToStr();
          // Use the flags of the merged Section for the segment
          const SegmentKey key(segmentName, (*msi)->flags());
          const std::pair<SegmentKey, Segment<target_endianness, is64Bits> *> 
                                                  currentSegment(key, nullptr);
          std::pair<typename SegmentMapT::iterator, bool> 
                              segmentInsert(_segmentMap.insert(currentSegment));

          if (!segmentInsert.second) {
            segment = segmentInsert.first->second;
          } else {
            segment = new (_allocator.Allocate
                           <Segment<target_endianness, is64Bits>>())
                           Segment<target_endianness, is64Bits>
                           (segmentName, getSegmentType(section), 
                            _options);
            segmentInsert.first->second = segment;
            _segments.push_back(segment);
          }
          segment->append(section);
        }
      }
    }
  }

  void addSection(Chunk<target_endianness, is64Bits> *c) {
    _sections.push_back(c);
  }

  void assignFileOffsets() {
    std::sort(_segments.begin(), 
              _segments.end(), 
              Segment<target_endianness, is64Bits>::compareSegments);
    int ordinal = 0;
    // Compute the number of segments that might be needed, so that the
    // size of the program header can be computed
    uint64_t offset = 0;
    for (auto si : _segments) {
      si->setOrdinal(++ordinal);
      si->assignOffsets(offset);
      offset += si->fileSize();
    }
  }

  void setELFHeader(ELFHeader<target_endianness, is64Bits> *e) {
    _elfHeader = e;
  }

  void setProgramHeader(ELFProgramHeader<target_endianness, is64Bits> *p) {
    _programHeader = p;
  }

  void assignVirtualAddress() {
    int32_t numSlices = 0;
    uint64_t virtualAddress = _options.baseAddress();

    // Add the ELF Header
    if (_elfHeader) {
      _elfHeader->setFileOffset(0);
      _elfHeader->setVAddr(virtualAddress);
    }
    // Add the program header
    if (_programHeader) {
      _programHeader->setVAddr(uint64_t(virtualAddress + _elfHeader->fileSize()));
      _programHeader->setFileOffset(_elfHeader->fileSize());
    }
    bool newSegmentHeaderAdded = true;
    while (true) {
      for (auto si : _segments) {
        newSegmentHeaderAdded = _programHeader->addSegment(si);
        numSlices += si->numSlices();
      }
      if (!newSegmentHeaderAdded) 
        break;
      uint64_t fileoffset = _elfHeader->fileSize() + _programHeader->fileSize();
      uint64_t address = virtualAddress;
      // Fix the offsets after adding the program header
      for (auto &si : _segments) {
        // Align the segment to a page boundary
        fileoffset = llvm::RoundUpToAlignment(fileoffset, _options.pageSize());
        si->assignOffsets(fileoffset);
        fileoffset = si->fileOffset() + si->fileSize();
      }
      // start assigning virtual addresses
      for (auto si = _segments.begin(); si != _segments.end(); ++si) {
        (*si)->setVAddr(virtualAddress);
        // The first segment has the virtualAddress set to the base address as
        // we have added the file header and the program header dont align the
        // first segment to the pagesize 
        (*si)->assignVirtualAddress(address, (si == _segments.begin()));
        (*si)->setMemSize(address - virtualAddress);
        virtualAddress = llvm::RoundUpToAlignment(address, _options.pageSize());
      }
      _programHeader->resetProgramHeaders();
    }
    Section<target_endianness, is64Bits> *section;
    // Fix the offsets of all the atoms within a section
    for (auto &si : _sections) {
      section = llvm::dyn_cast<Section<target_endianness, is64Bits>>(si);
      if (section && 
          DefaultELFLayout<target_endianness, is64Bits>::hasOutputSegment(section))
        section->assignOffsets(section->fileOffset());
    }
    // Set the size of the merged Sections
    for (auto msi = merged_sections_begin(), mse = merged_sections_end(); 
                                             msi != mse; ++msi) {
      uint64_t sectionfileoffset = 0;
      uint64_t startFileOffset = 0;
      uint64_t sectionsize = 0;
      bool isFirstSection = true;
      for (auto si = (*msi)->begin_sections(); si != (*msi)->end_sections(); ++si) {
        if (isFirstSection) {
          startFileOffset = (*si)->fileOffset();
          isFirstSection = false;
        }
        sectionfileoffset = (*si)->fileOffset();
        sectionsize = (*si)->fileSize();
      }
      sectionsize = (sectionfileoffset - startFileOffset) + sectionsize;
      (*msi)->setFileOffset(startFileOffset);
      (*msi)->setSize(sectionsize);
    }
    // Set the virtual addr of the merged Sections
    for (auto msi = merged_sections_begin(), mse = merged_sections_end(); 
                                             msi != mse; ++msi) {
      uint64_t sectionstartaddr = 0;
      uint64_t startaddr = 0;
      uint64_t sectionsize = 0;
      bool isFirstSection = true;
      for (auto si = (*msi)->begin_sections(), se = (*msi)->end_sections(); 
                                               si != se; ++si) {
        if (isFirstSection) {
          startaddr = (*si)->virtualAddr();
          isFirstSection = false;
        }
        sectionstartaddr = (*si)->virtualAddr();
        sectionsize = (*si)->memSize();
      }
      sectionsize = (sectionstartaddr - startaddr) + sectionsize;
      (*msi)->setMemSize(sectionsize);
      (*msi)->setAddr(startaddr);
    }
  }

  void assignOffsetsForMiscSections() {
    uint64_t fileoffset = 0;
    uint64_t size = 0;
    for (auto si : _segments) {
      fileoffset = si->fileOffset();
      size = si->fileSize();
    }
    fileoffset = fileoffset + size;
    Section<target_endianness, is64Bits> *section;
    for (auto si : _sections) {
      section = llvm::dyn_cast<Section<target_endianness, is64Bits>>(si);
      if (section && 
          DefaultELFLayout<target_endianness, is64Bits>::hasOutputSegment
                                                         (section))
        continue;
      fileoffset = llvm::RoundUpToAlignment(fileoffset, si->align2());
      si->setFileOffset(fileoffset);
      si->setVAddr(0);
      fileoffset += si->fileSize();
    }
  }

  void finalize() {
    for (auto &si : _sections) {
      si->finalize();
    }
  }

  bool findAtomAddrByName(const StringRef name, uint64_t &addr) {
    Section<target_endianness, is64Bits> *section;
    for (auto ai = _sections.begin(); ai != _sections.end(); ++ai) {
      if ((*ai)->kind() == Chunk<target_endianness, is64Bits>::K_ELFSection) {
        section = llvm::dyn_cast<Section<target_endianness, is64Bits>>(*ai);
        if (section->findAtomAddrByName(name, addr))
         return true;
      }
    }
    return false;    
  }

  MergedSectionIter merged_sections_begin() { 
    return _mergedSections.begin();
  }

  MergedSectionIter merged_sections_end() { 
    return _mergedSections.end();
  }

  ChunkIter sections_begin() {
    return _sections.begin();
  }
  ChunkIter sections_end() {
    return _sections.end();
  }

  ChunkIter segments_begin() {
    return _segments.begin();
  }

  ChunkIter segments_end() {
    return _segments.end();
  }

  ELFHeader<target_endianness, is64Bits> *elfHeader() { 
    return _elfHeader;
  }

  ELFProgramHeader<target_endianness, is64Bits> *elfProgramHeader() { 
    return _programHeader;
  }

private:
  SectionMapT _sectionMap;
  MergedSectionMapT _mergedSectionMap;
  SegmentMapT _segmentMap;

  std::vector<Chunk<target_endianness, is64Bits> *> _sections;
  std::vector<Segment<target_endianness, is64Bits> *> _segments;
  std::vector<MergedSections<target_endianness, is64Bits> *> _mergedSections;
  ELFHeader<target_endianness, is64Bits> *_elfHeader;
  ELFProgramHeader<target_endianness, is64Bits> *_programHeader;
  llvm::BumpPtrAllocator _allocator;
  const WriterOptionsELF &_options;
};

//===----------------------------------------------------------------------===//
//  ELFExecutableWriter Class
//===----------------------------------------------------------------------===//
template<support::endianness target_endianness, bool is64Bits>
class ELFExecutableWriter : public ELFWriter {
public:
  typedef object::Elf_Shdr_Impl<target_endianness, is64Bits> Elf_Shdr;
  typedef object::Elf_Sym_Impl<target_endianness, is64Bits> Elf_Sym;

  ELFExecutableWriter(const WriterOptionsELF &options);

private:
  // build the sections that need to be created 
  void buildChunks(const lld::File &file);
  virtual error_code writeFile(const lld::File &File, StringRef path);
  void buildAtomToAddressMap();
  void buildSymbolTable ();
  void buildSectionHeaderTable();
  void assignSectionsWithNoSegments();
  void addAbsoluteUndefinedSymbols(const lld::File &File);

  uint64_t addressOfAtom(const Atom *atom) { 
    return _atomToAddressMap[atom];
  }

  KindHandler *kindHandler() { return _referenceKindHandler.get(); }

  void createDefaultSections();

  const WriterOptionsELF &_options;

  typedef llvm::DenseMap<const Atom*, uint64_t> AtomToAddress;
  std::unique_ptr<KindHandler> _referenceKindHandler;
  AtomToAddress _atomToAddressMap;
  llvm::BumpPtrAllocator _chunkAllocate;
  DefaultELFLayout<target_endianness, is64Bits> *_layout;
  ELFHeader<target_endianness, is64Bits> *_elfHeader;
  ELFProgramHeader<target_endianness, is64Bits> *_programHeader;
  ELFSymbolTable<target_endianness, is64Bits> * _symtab;
  ELFStringTable<target_endianness, is64Bits> *_strtab;
  ELFStringTable<target_endianness, is64Bits> *_shstrtab;
  ELFSectionHeader<target_endianness, is64Bits> *_shdrtab;
};

//===----------------------------------------------------------------------===//
//  ELFExecutableWriter
//===----------------------------------------------------------------------===//
template<support::endianness target_endianness, bool is64Bits>
ELFExecutableWriter<target_endianness, is64Bits>
         ::ELFExecutableWriter(const WriterOptionsELF &options)
  : _options(options)
  , _referenceKindHandler(KindHandler::makeHandler(_options.machine(),
                                                   target_endianness))
{
  _layout = new DefaultELFLayout<target_endianness, is64Bits>(options);
}

template<support::endianness target_endianness, bool is64Bits>
void ELFExecutableWriter<target_endianness, is64Bits>
                        ::buildChunks(const lld::File &file){
  for (const DefinedAtom *definedAtom : file.defined() ) {
    _layout->addAtom(definedAtom);
  }
}

template<support::endianness target_endianness, bool is64Bits>
void ELFExecutableWriter<target_endianness, is64Bits>::buildSymbolTable () {
  Section<target_endianness, is64Bits> *section;
  for (auto si = _layout->sections_begin(); si != _layout->sections_end(); ++si) {
    if ((*si)->kind() != Chunk<target_endianness, is64Bits>::K_ELFSection) 
      continue;
    section = llvm::dyn_cast<Section<target_endianness, is64Bits>>(*si);
    for (auto ai = section->atoms_begin(); ai != section->atoms_end(); ++ai) {
      _symtab->addSymbol(ai->first, section->ordinal(), ai->second.second);
    }
  }
}

template<support::endianness target_endianness, bool is64Bits>
void ELFExecutableWriter<target_endianness, is64Bits>::
                addAbsoluteUndefinedSymbols(const lld::File &file) {
 for (const UndefinedAtom *a : file.undefined()) {
   _symtab->addSymbol(a, ELF::SHN_UNDEF);
 }
 
 for (const AbsoluteAtom *a : file.absolute()) {
   _symtab->addSymbol(a, ELF::SHN_ABS);
 }
}

template<support::endianness target_endianness, bool is64Bits>
void ELFExecutableWriter<target_endianness, is64Bits>
              ::buildAtomToAddressMap () {
  Section<target_endianness, is64Bits> *section;
  for (auto si = _layout->sections_begin(); 
       si != _layout->sections_end(); ++si) {
    if ((*si)->kind() != Chunk<target_endianness, is64Bits>::K_ELFSection) 
      continue;
    section = llvm::dyn_cast<Section<target_endianness, is64Bits>>(*si);
    for (auto ai = section->atoms_begin(); ai != section->atoms_end(); ++ai) {
      _atomToAddressMap[ai->first] = (ai)->second.second;
    }
  }
}

template<support::endianness target_endianness, bool is64Bits>
void ELFExecutableWriter<target_endianness, is64Bits>
              ::buildSectionHeaderTable() {
  for (auto msi = _layout->merged_sections_begin(); 
       msi != _layout->merged_sections_end(); ++msi) {
    if ((*msi)->kind() != Chunk<target_endianness, is64Bits>::K_ELFSection) 
      continue;
    if ((*msi)->hasSegment())
      _shdrtab->appendSection(*msi);
  }
}

template<support::endianness target_endianness, bool is64Bits>
void ELFExecutableWriter<target_endianness, is64Bits>
              ::assignSectionsWithNoSegments() {
  Section<target_endianness, is64Bits> *section;
  for (auto msi = _layout->merged_sections_begin(); 
       msi != _layout->merged_sections_end(); ++msi) {
    if ((*msi)->kind() != Chunk<target_endianness, is64Bits>::K_ELFSection) 
      continue;
    if (!(*msi)->hasSegment())
      _shdrtab->appendSection(*msi);
  }
  _layout->assignOffsetsForMiscSections();
  for (auto si = _layout->sections_begin(); 
       si != _layout->sections_end(); ++si) {
    if ((*si)->kind() != Chunk<target_endianness, is64Bits>::K_ELFSection) 
      continue;
    section = llvm::dyn_cast<Section<target_endianness, is64Bits>>(*si);
    if (!DefaultELFLayout<target_endianness, is64Bits>::hasOutputSegment
                                                        (section))
      _shdrtab->updateSection(section);
  }
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFExecutableWriter<target_endianness, is64Bits>
                    ::writeFile(const lld::File &file, StringRef path) {
  buildChunks(file);
  // Create the default sections like the symbol table, string table, and the
  // section string table
  createDefaultSections();

  // Set the Layout
  _layout->assignSectionsToSegments();
  _layout->assignFileOffsets();
  _layout->assignVirtualAddress();

  // Build the Atom To Address map for applying relocations
  buildAtomToAddressMap();

  // Create symbol table and section string table
  buildSymbolTable();

  // add other symbols
  addAbsoluteUndefinedSymbols(file);

  // Finalize the layout by calling the finalize() functions
  _layout->finalize();

  // build Section Header table
  buildSectionHeaderTable();

  // assign Offsets and virtual addresses
  // for sections with no segments
  assignSectionsWithNoSegments();

  uint64_t totalSize = _shdrtab->fileOffset() + _shdrtab->fileSize();

  OwningPtr<FileOutputBuffer> buffer;
  error_code ec = FileOutputBuffer::create(path,
                                           totalSize, buffer,
                                           FileOutputBuffer::F_executable);
  if (ec)
    return ec;

  for (auto si = _layout->sections_begin(); si != _layout->sections_end(); ++si)
    (*si)->write(this, buffer);

  _elfHeader->e_ident(ELF::EI_CLASS, (_options.is64Bit() ? ELF::ELFCLASS64
                                                        : ELF::ELFCLASS32));
  _elfHeader->e_ident(ELF::EI_DATA, (_options.endianness() == llvm::support::big)
                                    ? ELF::ELFDATA2MSB : ELF::ELFDATA2LSB);
  _elfHeader->e_ident(ELF::EI_VERSION, 1);
  _elfHeader->e_ident(ELF::EI_OSABI, 0);
  _elfHeader->e_type(_options.type());
  _elfHeader->e_machine(_options.machine());
  _elfHeader->e_version(1);
  _elfHeader->e_entry(0ULL);
  _elfHeader->e_phoff(_programHeader->fileOffset());
  _elfHeader->e_shoff(_shdrtab->fileOffset());
  _elfHeader->e_phentsize(_programHeader->entsize());
  _elfHeader->e_phnum(_programHeader->numHeaders());
  _elfHeader->e_shentsize(_shdrtab->entsize());
  _elfHeader->e_shnum(_shdrtab->numHeaders());
  _elfHeader->e_shstrndx(_shstrtab->ordinal());
  uint64_t virtualAddr = 0;
  _layout->findAtomAddrByName("_start", virtualAddr);
  _elfHeader->e_entry(virtualAddr);
  _elfHeader->write(this, buffer);
  _programHeader->write(this, buffer);

  return buffer->commit();
}

template<support::endianness target_endianness, bool is64Bits>
void ELFExecutableWriter<target_endianness, is64Bits>
                    ::createDefaultSections() {
  _elfHeader = new ELFHeader<target_endianness, is64Bits>();
  _programHeader = new ELFProgramHeader<target_endianness, is64Bits>();
  _layout->setELFHeader(_elfHeader);
  _layout->setProgramHeader(_programHeader);

  _symtab = new ELFSymbolTable<target_endianness, is64Bits>
            (".symtab", 
            DefaultELFLayout<target_endianness, is64Bits>::ORDER_SYMBOL_TABLE);
  _strtab = new ELFStringTable<target_endianness, is64Bits>
          (".strtab", 
          DefaultELFLayout<target_endianness, is64Bits>::ORDER_STRING_TABLE);
  _shstrtab = new ELFStringTable<target_endianness, is64Bits>
          (".shstrtab", 
          DefaultELFLayout<target_endianness, is64Bits>::ORDER_SECTION_STRINGS);
  _shdrtab  = new ELFSectionHeader<target_endianness, is64Bits>
         (DefaultELFLayout<target_endianness, is64Bits>::ORDER_SECTION_HEADERS);
  _layout->addSection(_symtab);
  _layout->addSection(_strtab);
  _layout->addSection(_shstrtab);
  _shdrtab->setStringSection(_shstrtab);
  _symtab->setStringSection(_strtab);
  _layout->addSection(_shdrtab);
}

} // namespace elf

Writer *createWriterELF(const WriterOptionsELF &options) {

  // Set the default layout to be the static executable layout
  // We would set the layout to a dynamic executable layout 
  // if we came across any shared libraries in the process
  
  if (!options.is64Bit() && options.endianness() == llvm::support::little)
    return new lld::elf::ELFExecutableWriter<support::little, false>(options);
  else if (options.is64Bit() && options.endianness() == llvm::support::little)
    return new lld::elf::ELFExecutableWriter<support::little, true>(options);
  else if (!options.is64Bit() && options.endianness() == llvm::support::big)
    return new lld::elf::ELFExecutableWriter<support::big, false>(options);
  else if (options.is64Bit() && options.endianness() == llvm::support::big)
    return new lld::elf::ELFExecutableWriter<support::big, true>(options);

  llvm_unreachable("Invalid Options!");
}
} // namespace lld
