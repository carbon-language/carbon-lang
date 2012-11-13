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
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

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

#include <map>
#include <tuple>
#include <vector>
#include <algorithm>

using namespace llvm;
using namespace llvm::object;
namespace lld {
namespace elf {

// The group decides where sections of similar permissions reside
// inside a file. This is also used to create program headers. Each group
// will be in a new segment.

/// \name Chunk Groups
/// Each "Chunk" should be arranged in a weak strict order
/// This is done to minimize memory utilization when creating program segments.
/// This will remain in place till a more felxible mechanism of moving chunks 
/// around and packing for memory efficiency is put in place. 
/// The Chunks are arranged by group numbers. Every switch from group A to B
/// triggers a formation of new segment. Additionally segments are also made
/// within a group if there are vast regions (multiple pages) of 0 for 
/// alignment constraints.
enum {
/// @{
///Invalid group (default on creation)
  CG_INVALID = 0x0,
///This is a "header" kind of chunk that goes at beginning of ELF file
  CG_HEADER = 0x1,
///This is a section chunk with read write and execute permissions
  CG_RWX = 0x3,
///This is a section chunk with read and execute permissions
  CG_RX = 0x4,
///This is a section chunk with read permission
  CG_R = 0x5,
///This is a section chunk with read and write permissions
  CG_RW = 0x6,
///This is a section chunk with write and execute permissions
  CG_WX = 0x7,
///This is a section chunk with write permission
  CG_W = 0x8,
///This is a section chunk with execute permission
  CG_X = 0x9,
///This is a section which is no loaded by program loader
  CG_NO_LOAD = 0xFE,
///This is ELF file metadata that goes at end of file
  CG_FILE_END = 0xFF 
/// @}
};

template<support::endianness target_endianness, bool is64Bits>
class ELFWriter;

/// \brief A Chunk is a contiguous range of space.
template<support::endianness target_endianness, bool is64Bits>
class Chunk {
public:
  LLVM_ELF_IMPORT_TYPES(target_endianness, is64Bits)
  virtual             ~Chunk() {}
  virtual StringRef   segmentName() const = 0;
  virtual bool        occupiesNoDiskSpace();
  virtual void        write(uint8_t *fileBuffer) = 0;
  void                assignFileOffset(uint64_t &curOff, uint64_t &curAddr);
  virtual const char *info() = 0;
  uint64_t            size() const;
  uint64_t            address() const;
  uint64_t            fileOffset() const;
  uint64_t            align2() const;
  static uint64_t     alignTo(uint64_t value, uint8_t align2);
  uint64_t            ordinal() const { return _ordinal;}
  void                setOrdinal(uint64_t newVal) { _ordinal = newVal;}
  void                setGroup(uint16_t val) { _group = val;}
  uint16_t            group() const { return _group;}
  void                init();
  bool                isLoadable() { return _isLoadable; }
  void                isLoadable(uint64_t val) {  _isLoadable = val; }
  enum class Kind {
    Header,      // This is a header chunk
    Section      // chunk represents a section
  };
  Kind getChunkKind() const { return _cKind; }
private:
  const Kind _cKind;
protected:
  Chunk();
  Chunk(Kind K): _cKind(K) { this->init();}
  uint64_t _size;
  uint64_t _address;
  uint64_t _fileOffset;
  uint64_t _align2;
  uint64_t _ordinal;
  uint16_t _group;
  bool     _isLoadable;
};

template<support::endianness target_endianness, bool is64Bits>
static void swapChunkPositions(Chunk<target_endianness, is64Bits>&a,
                               Chunk<target_endianness, is64Bits>&b) {
  uint64_t tempOrdinal;
  if (a.ordinal() == b.ordinal()) return;
  tempOrdinal = a.ordinal();
  a.setOrdinal(b.ordinal());
  b.setOrdinal(tempOrdinal);
}

template<support::endianness target_endianness, bool is64Bits>
bool chunkGroupSort(Chunk<target_endianness, is64Bits> *A,
                    Chunk<target_endianness, is64Bits> *B) {
  if (A->group() == CG_INVALID || B->group() == CG_INVALID)
    llvm_unreachable("Invalid group number");
  return A->group() < B->group();
}

template<support::endianness target_endianness, bool is64Bits>
struct ChunkComparator {
  bool operator()(uint16_t A, Chunk<target_endianness, is64Bits> *B) {
    return A < B->group();
  }
  bool operator()(Chunk<target_endianness, is64Bits> *A, uint16_t B) {
    return A->group() < B;
  }
#if defined(_ITERATOR_DEBUG_LEVEL) && _ITERATOR_DEBUG_LEVEL == 2
 bool operator()(Chunk<target_endianness, is64Bits> *A,
        Chunk<target_endianness, is64Bits> *B) {
   return A->group() < B->group();
 }
#endif
};

/// Pair of atom and offset in section.
typedef std::tuple<const DefinedAtom*, uint64_t> AtomInfo;

/// \brief A SectionChunk represents ELF sections 
template<support::endianness target_endianness, bool is64Bits>
class SectionChunk : public Chunk<target_endianness, is64Bits> {
public:
  virtual StringRef   segmentName() const { return _segmentName; }
  virtual bool        occupiesNoDiskSpace();
  virtual const char  *info();
  StringRef           sectionName() { return _sectionName; }
  uint64_t            shStrtableOffset(){ return _offsetInStringTable; }
  void                setShStrtableOffset (uint64_t val) {
                       _offsetInStringTable = val; } 
  uint32_t            flags()  { return _flags; }
  uint32_t            type() const { return _type; }
  uint64_t            link()   { return _link; }
  void                link(uint64_t val)   { _link = val; }
  uint16_t            shinfo() { return _shinfo; }
  uint64_t            entsize() { return _entsize; }
  SectionChunk(StringRef secName, StringRef segName, bool loadable, 
               uint64_t flags , uint64_t link,  uint64_t info ,
               uint64_t type, uint64_t entsz, const WriterOptionsELF &op, 
               ELFWriter<target_endianness, is64Bits> &writer);

  static inline bool classof(const Chunk<target_endianness, is64Bits> *c) {
    return c->getChunkKind() == Chunk<target_endianness, is64Bits>::Kind
                                                                  ::Section;
  }
protected:
  uint64_t                                _link;
  uint64_t                                _shinfo;
  uint16_t                                _entsize;
  StringRef                               _segmentName;
  StringRef                               _sectionName;
  const WriterOptionsELF                 &_options;
  ELFWriter<target_endianness, is64Bits> &_writer;
  uint64_t                                _flags;
  uint64_t                                _type;
  uint64_t                                _offsetInStringTable;
};

template<support::endianness target_endianness, bool is64Bits>
bool IsBss(const  Chunk<target_endianness, is64Bits> *A) {
  if (auto X = llvm::dyn_cast<SectionChunk<target_endianness, is64Bits>>(A)) 
    return X->type() != ELF::SHT_NOBITS;
  llvm_unreachable("Call to a non section type chunk");
  // adding a non-reachable return bool for making compiler happy
  return false;
}

/// \brief A StockSectionChunk is a section created by linker with all 
///        attributes concluded from the defined atom contained within.
template<support::endianness target_endianness, bool is64Bits>
class StockSectionChunk : public SectionChunk<target_endianness, is64Bits> {
public:
  virtual StringRef segmentName() const { return this->_segmentName; }
  void                appendAtom(const DefinedAtom*);
  virtual void        write(uint8_t *filebuffer);
  const               ArrayRef<AtomInfo> atoms() const;
  StockSectionChunk(StringRef sectionName, bool loadable,
                    DefinedAtom::ContentType type, 
                    const WriterOptionsELF &options,
                    ELFWriter<target_endianness, is64Bits> &writer);
private:
  std::vector<AtomInfo>                   _atoms;
};
                        
/// \brief An ELFHeaderChunk represents the Elf[32/64]_Ehdr structure at the
///        start of an ELF executable file.
template<support::endianness target_endianness, bool is64Bits>
class ELFHeaderChunk : public Chunk<target_endianness, is64Bits> {
public:
  LLVM_ELF_IMPORT_TYPES(target_endianness, is64Bits)
  typedef object::Elf_Ehdr_Impl<target_endianness, is64Bits> Elf_Ehdr;

  ELFHeaderChunk(const WriterOptionsELF &options,
                 const File &file);

  void e_ident(int I, unsigned char C) { _eh.e_ident[I] = C; }
  void e_type(uint16_t type)           { _eh.e_type = type; }
  void e_machine(uint16_t machine)     { _eh.e_machine = machine; }
  void e_version(uint32_t version)     { _eh.e_version = version; }
  void e_entry(uint64_t entry)         { _eh.e_entry = entry; }
  void e_phoff(uint64_t phoff)         { _eh.e_phoff = phoff; }
  void e_shoff(uint64_t shoff)         { _eh.e_shoff = shoff; }
  void e_flags(uint32_t flags)         { _eh.e_flags = flags; }
  void e_ehsize(uint16_t ehsize)       { _eh.e_ehsize = ehsize; }
  void e_phentsize(uint16_t phentsize) { _eh.e_phentsize = phentsize; }
  void e_phnum(uint16_t phnum)         { _eh.e_phnum = phnum; }
  void e_shentsize(uint16_t shentsize) { _eh.e_shentsize = shentsize; }
  void e_shnum(uint16_t shnum)         { _eh.e_shnum = shnum; }
  void e_shstrndx(uint16_t shstrndx)   { _eh.e_shstrndx = shstrndx; }
  uint64_t  size()                     { return sizeof (Elf_Ehdr); }

  virtual StringRef   segmentName() const;
  virtual void        write(uint8_t *fileBuffer);
  virtual const char *info();
  static inline bool classof(const Chunk<target_endianness, is64Bits> *c) {
    return c->getChunkKind() ==  Chunk<target_endianness, is64Bits>::Kind
                                                                   ::Header;
  }

private:
  Elf_Ehdr             _eh;
};

/// \brief An ELFSectionHeaderChunk represents the Elf[32/64]_Shdr structure
///        that is placed right after the ELFHeader.
///
template<support::endianness target_endianness, bool is64Bits>
class ELFSectionHeaderChunk : public Chunk<target_endianness, is64Bits> {
public:
  LLVM_ELF_IMPORT_TYPES(target_endianness, is64Bits)
  typedef object::Elf_Shdr_Impl<target_endianness, is64Bits> Elf_Shdr;
  ELFSectionHeaderChunk(const WriterOptionsELF &Options,
                        ELFWriter<target_endianness, is64Bits>&);
  void createHeaders();
  virtual StringRef   segmentName() const;
  virtual void        write(uint8_t *filebuffer);
  virtual const char *info();
  void                computeSize();
  uint16_t            count();
  uint16_t            size();
  void                fixOffsets();
  const ArrayRef<Elf_Shdr*> sectionInfo() {
    return _sectionInfo;
  }
  static inline bool classof(const Chunk<target_endianness, is64Bits> *c) {
    return c->getChunkKind() ==  Chunk<target_endianness, is64Bits>::Kind
                                                                   ::Header;
  }

private:
  const WriterOptionsELF                 &_options;
  ELFWriter<target_endianness, is64Bits> &_writer;
  llvm::BumpPtrAllocator                  _sectionAllocate;
  std::vector<Elf_Shdr*>                  _sectionInfo;
};

/// \brief Represents the shstr section.
///
/// This is a contiguous memory that has all the symbol strings each ending with
/// null character. We might need more than one such chunks shstrtab for setting
/// e_shstrndx in ELHHeaderChunk and strtab for use with symtab
template<support::endianness target_endianness, bool is64Bits>
class ELFStringSectionChunk : public SectionChunk<target_endianness, is64Bits> {
public:
  ELFStringSectionChunk(const WriterOptionsELF &Options,
                        ELFWriter<target_endianness, is64Bits> &writer,
                        StringRef secName);
  virtual StringRef   segmentName() const { return this->_segmentName; }
  uint64_t            addString(StringRef symName);
  const char          *info();
  virtual void        write(uint8_t *filebuffer);

private:
  std::vector<StringRef> _stringSection;
};

/// \brief Represents the symtab section
/// 
/// ELFSymbolTableChunk represents the Symbol table as per ELF ABI
/// This is a table with Elf[32/64]_Sym entries in it. 
template<support::endianness target_endianness, bool is64Bits>
class ELFSymbolTableChunk : public SectionChunk<target_endianness, is64Bits> {
public:
  typedef object::Elf_Sym_Impl<target_endianness, is64Bits> Elf_Sym;
  ELFSymbolTableChunk(const WriterOptionsELF &options,
                      ELFWriter<target_endianness, is64Bits> &writer,
                      StringRef secName);
  virtual StringRef   segmentName() const { return this->_segmentName; }
  void                addSymbol(const Atom *a, uint16_t shndx);
  void                addSymbol(Elf_Sym *x); 
  void                fixSymbolValue(); 
  const char          *info();
  void                setAttributes();
  virtual void               write(uint8_t *fileBuffer);

private:
  std::vector<Elf_Sym*> _symbolTable;
  ELFStringSectionChunk<target_endianness, is64Bits> *_stringSection;
  llvm::BumpPtrAllocator _symbolAllocate;
  std::map<Elf_Sym*, const Atom*> _symbolToAtom;
};

/// \brief  ELFProgramHeaderChunk represents the Elf[32/64]_Phdr structure near
/// the start of an ELF executable file. ELFHeader's e_phentsize and e_phnum 
/// show the number of entries in table and size of each one.
template<support::endianness target_endianness, bool is64Bits>
class ELFProgramHeaderChunk : public Chunk<target_endianness, is64Bits> {
public:
  LLVM_ELF_IMPORT_TYPES(target_endianness, is64Bits)
  typedef object::Elf_Phdr<target_endianness, is64Bits> Elf_Phdr;
  ELFProgramHeaderChunk(const WriterOptionsELF &options, 
                        ELFWriter<target_endianness, is64Bits> &);

  virtual StringRef   segmentName() const;
  virtual void        write(uint8_t *filebuffer);
  virtual const char *info();
  void                createPHeaders();
  uint64_t            computeNumber();
  static inline bool classof(const Chunk<target_endianness, is64Bits> *c) {
    return c->getChunkKind() ==  Chunk<target_endianness, is64Bits>::Kind
                                                                   ::Header;
  }

private:
  typedef typename std::vector<SectionChunk<target_endianness, is64Bits>*>
                    ::iterator secIterator;
  const WriterOptionsELF                 &_options;
  ELFWriter<target_endianness, is64Bits> &_writer;
  std::vector<Elf_Phdr*>                 _programHeaders;
  llvm::BumpPtrAllocator                 _headerAllocate;
  typedef std::pair<uint64_t, uint64_t>  sectionRange; 
  std::vector<sectionRange>              _segments;
  std::map<uint16_t, uint16_t>           _groupToPF;

};

//===----------------------------------------------------------------------===//
//  Chunk
//===----------------------------------------------------------------------===//

template<support::endianness target_endianness, bool is64Bits>
Chunk<target_endianness, is64Bits>::Chunk(){
  this->init();
}

template<support::endianness target_endianness, bool is64Bits>
void Chunk<target_endianness, is64Bits>::init(){
  this->_size = 0; 
  this->_address = 0; 
  this->_fileOffset = 0; 
  this->_align2 = 0; 
  this->_group = CG_INVALID; 
  this->_isLoadable = false;
   // 0 and 1 are reserved. 0 for ELF header and 1 for program  header.
  static uint64_t orderNumber = 0;
  _ordinal = orderNumber++;
}

template<support::endianness target_endianness, bool is64Bits>
bool Chunk<target_endianness, is64Bits>::occupiesNoDiskSpace() {
  return false;
}

template<support::endianness target_endianness, bool is64Bits>
uint64_t Chunk<target_endianness, is64Bits>::size() const {
  return _size;
}

template<support::endianness target_endianness, bool is64Bits>
uint64_t Chunk<target_endianness, is64Bits>::align2() const {
  return _align2;
}

template<support::endianness target_endianness, bool is64Bits>
uint64_t Chunk<target_endianness, is64Bits>::address() const {
  return _address;
}

template<support::endianness target_endianness, bool is64Bits>
uint64_t Chunk<target_endianness, is64Bits>::fileOffset() const {
  return _fileOffset;
}

template<support::endianness target_endianness, bool is64Bits>
uint64_t Chunk<target_endianness, is64Bits>::
         alignTo(uint64_t value, uint8_t align2) {
  uint64_t align = 1 << align2;
  return (value + (align - 1)) & (-align);
}

template<support::endianness target_endianness, bool is64Bits>
void Chunk<target_endianness, is64Bits>::
     assignFileOffset(uint64_t &curOffset, uint64_t &curAddress) {
  if (occupiesNoDiskSpace()) {
    // FileOffset does not change, but virtual address does change.
    uint64_t alignedAddress =
      alignTo(curAddress, _align2 ? static_cast<uint8_t>(llvm::Log2_64(_align2))
                                  : 0);
    _address = alignedAddress;
    curAddress = alignedAddress + _size;
  } else {
    // FileOffset and address both move by _size amount after alignment.
    uint64_t alignPadding =
      alignTo(curAddress, _align2 ? static_cast<uint8_t>(llvm::Log2_64(_align2))
                                  : 0) - curAddress;
    _fileOffset = curOffset + alignPadding;
    _address    = curAddress + alignPadding;
    curOffset   = _fileOffset + _size;
    curAddress  = _address + _size;
  }

  DEBUG_WITH_TYPE("WriterELF-layout", dbgs()  
                      << "   fileOffset="
                      << format("0x%08X", _fileOffset)
                      << " address="
                      << format("0x%016X", _address)
                      << " info=" << info() << "\n");
}

//===----------------------------------------------------------------------===//
//  SectionChunk
//===----------------------------------------------------------------------===//

template<support::endianness target_endianness, bool is64Bits>
SectionChunk<target_endianness, is64Bits>::
 SectionChunk(StringRef secName, StringRef segName, bool loadable, 
              uint64_t flags , uint64_t link,  uint64_t info , uint64_t type,
              uint64_t entsz, const WriterOptionsELF &op, 
              ELFWriter<target_endianness, is64Bits> &writer)
  : Chunk<target_endianness, is64Bits>(Chunk<target_endianness, is64Bits>
                                       ::Kind::Section)
  , _link(link)
  , _shinfo(info)
  , _entsize(entsz)
  , _segmentName(segName)
  , _sectionName(secName)
  , _options(op)
  , _writer(writer)
  , _flags(flags)
  , _type(type)
  , _offsetInStringTable(0) {
    this->isLoadable(loadable);
}
//FIXME: We need to make decision here for every section created
template<support::endianness target_endianness, bool is64Bits>
bool SectionChunk<target_endianness, is64Bits>::occupiesNoDiskSpace() {
  return false;
}

template<support::endianness target_endianness, bool is64Bits>
const char *SectionChunk<target_endianness, is64Bits>::info() {
  return _sectionName.data();
}

//===----------------------------------------------------------------------===//
//  StockSectionChunk
//===----------------------------------------------------------------------===//

template<support::endianness target_endianness, bool is64Bits>
StockSectionChunk<target_endianness, is64Bits>::
  StockSectionChunk(StringRef secName, bool loadable, 
                    DefinedAtom::ContentType type,
                    const WriterOptionsELF &options, 
                    ELFWriter<target_endianness, is64Bits> &writer)
 : SectionChunk<target_endianness, is64Bits>(secName, "PT_NULL", 
                                             loadable, 0lu, 0lu, 0u, 0lu, 0lu,
                                             options, writer) {
  this->_segmentName = this->_isLoadable ? "PT_LOAD" : "PT_NULL" ;
  // If the section is custom loadable section, group should be set explicitly.
  // Stock non loadable section go as NO_LOAD and others  will get their 
  // group determined by the atoms contained within. Many constant 
  // sections will have no symbols but the constants are referred as 
  // offset from start symbol, hence there may not be any defined atoms being 
  // appended to them, we force them  at time of creation to CG_R
  this->setGroup(this->isLoadable() ? CG_INVALID : CG_NO_LOAD);
  switch(type) {
  case DefinedAtom::typeCode:
    this->_type        = ELF::SHT_PROGBITS;
    break;
  case DefinedAtom::typeData:
    this->_type        = ELF::SHT_PROGBITS;
    break;
  case DefinedAtom::typeZeroFill:
    this->_type        = ELF::SHT_NOBITS;
    break;
  case DefinedAtom::typeConstant:
    this->_type        = ELF::SHT_PROGBITS;
    this->_flags       = ELF::SHF_ALLOC;
    this->setGroup(CG_R);
    break;
  default:
    llvm_unreachable("Unhandled content type for section!");
  }
}
 

template<support::endianness target_endianness, bool is64Bits>
const ArrayRef<AtomInfo> StockSectionChunk<target_endianness, is64Bits>::
    atoms() const {
  return _atoms;
}

template<support::endianness target_endianness, bool is64Bits>
void StockSectionChunk<target_endianness, is64Bits>::
     appendAtom(const DefinedAtom *atom) {
  static uint16_t groupArray[] = {CG_INVALID, CG_W, CG_R, CG_RW, CG_X,
                                  CG_WX, CG_RX, CG_RWX};
  // Figure out offset for atom in this section given alignment constraints.
  uint64_t offset = this->_size;
  DefinedAtom::Alignment atomAlign = atom->alignment();
  uint64_t align2 = 1 << atomAlign.powerOf2;
  uint64_t requiredModulus = atomAlign.modulus;
  uint64_t currentModulus = (offset % align2);
  if (currentModulus != requiredModulus) {
    if (requiredModulus > currentModulus)
      offset += requiredModulus - currentModulus;
    else
      offset += align2 + requiredModulus - currentModulus;
  }
  // Record max alignment of any atom in this section.
  if (align2 > this->_align2)
    this->_align2 = align2;
  // Assign atom to this section with this offset.
  _atoms.emplace_back(atom, offset);
  // Update section size to include this atom.
  this->_size = offset + atom->size();
  // Update permissions
  DefinedAtom::ContentPermissions perms = atom->permissions();
 
  if ((perms & DefinedAtom::permR__) == DefinedAtom::permR__)
    this->_flags |= ELF::SHF_ALLOC;
  if ((perms & DefinedAtom::permRW_) == DefinedAtom::permRW_)
    this->_flags |= (ELF::SHF_ALLOC | ELF::SHF_WRITE);
  if ((perms & DefinedAtom::permR_X) == DefinedAtom::permR_X)
    this->_flags |= (ELF::SHF_ALLOC | ELF::SHF_EXECINSTR);
  if (atom->contentType() == DefinedAtom::typeZeroFill)
    this->_flags |= (ELF::SHF_ALLOC | ELF::SHF_WRITE);
  this->setGroup(groupArray[this->_flags]);
}

template<support::endianness target_endianness, bool is64Bits>
void StockSectionChunk<target_endianness, is64Bits>
                      ::write(uint8_t *chunkBuffer) {
  // Each section's content is just its atoms' content.
  for (const auto &ai : _atoms ) {
    // Copy raw content of atom to file buffer.
   ArrayRef<uint8_t> content = std::get<0>(ai)->rawContent();
   uint64_t contentSize = content.size();
   if (contentSize == 0)
     continue;
   uint8_t *atomContent = chunkBuffer + std::get<1>(ai);
   std::copy_n(content.data(), contentSize, atomContent);

  for (const Reference *ref : *std::get<0>(ai)){
    uint32_t offset = ref->offsetInAtom();
    uint64_t targetAddress = 0;

    if ( ref->target() != nullptr )
      targetAddress = this->_writer.addressOfAtom(ref->target());

    uint64_t fixupAddress = this->_writer.addressOfAtom(std::get<0>(ai)) +
                            offset;
    this->_writer.kindHandler()->applyFixup(ref->kind(), ref->addend(),
                                            &atomContent[offset],
                                            fixupAddress,
                                            targetAddress);
    }
  }
}

//===----------------------------------------------------------------------===//
//  ELFStringSectionChunk
//===----------------------------------------------------------------------===//
template<support::endianness target_endianness, bool is64Bits>
ELFStringSectionChunk<target_endianness, is64Bits>::
    ELFStringSectionChunk(const WriterOptionsELF &options,
                          ELFWriter<target_endianness, is64Bits> &writer, 
                          StringRef secName) 
  : SectionChunk<target_endianness, is64Bits>(secName, "PT_NULL", 
                                              false, 0lu, 0lu, 0lu, 
                                              ELF::SHT_STRTAB, 0lu, options, 
                                              writer) {
  // First Add a null character. It also occupies 1 byte
  _stringSection.emplace_back("");
  this->_size = 1;
  this->setGroup(CG_NO_LOAD);
}

template<support::endianness target_endianness, bool is64Bits>
uint64_t ELFStringSectionChunk<target_endianness, is64Bits>::
         addString(StringRef symName) {
  _stringSection.emplace_back(symName);
  uint64_t offset = this->_size;
  this->_size += symName.size() + 1;

  return offset;
}

// We need to unwrap the _stringSection and then make one large memory 
// chunk of null terminated strings
template<support::endianness target_endianness, bool is64Bits>
void ELFStringSectionChunk<target_endianness, is64Bits>::
     write(uint8_t *chunkBuffer) {
  uint64_t chunkOffset = 0;
 
  for (auto it : _stringSection) {
    ::memcpy(chunkBuffer + chunkOffset, it.data(), it.size());
    chunkOffset += it.size();
    ::memcpy(chunkBuffer + chunkOffset, "", 1);
    chunkOffset += 1;
  }
}

template<support::endianness target_endianness, bool is64Bits>
const char *ELFStringSectionChunk<target_endianness, is64Bits>::info() {
  return "String Table";
}

//===----------------------------------------------------------------------===//
//  ELFSymbolTableChunk
//===----------------------------------------------------------------------===//
template< support::endianness target_endianness, bool is64Bits>
ELFSymbolTableChunk<target_endianness, is64Bits>::ELFSymbolTableChunk
                    (const WriterOptionsELF &options, 
                     ELFWriter<target_endianness, is64Bits> &writer, 
                     StringRef secName)
  : SectionChunk<target_endianness, is64Bits>(secName, StringRef("PT_NULL"), 
                                              false, 0, 0, 0, ELF::SHT_SYMTAB,
                                              sizeof(Elf_Sym), options, writer)
{
  _stringSection = this->_writer.strtab();
  Elf_Sym *symbol = new (_symbolAllocate.Allocate<Elf_Sym>()) Elf_Sym;
  memset ((void *)symbol,0, sizeof(Elf_Sym));
  _symbolTable.push_back(symbol);
  this->_link = 0;
  this->_entsize = sizeof(Elf_Sym);
  this->_size = sizeof(Elf_Sym);
  this->_align2 = this->_options.pointerWidth();
  this->setGroup(CG_NO_LOAD);
}

template< support::endianness target_endianness, bool is64Bits>
void ELFSymbolTableChunk<target_endianness, is64Bits>::fixSymbolValue(){
  for (auto sym : _symbolTable) {
    if ( sym->st_shndx != ELF::SHN_ABS) {
      sym->st_value = this->_writer.addressOfAtom(_symbolToAtom[sym]);
    }
  }
}

template< support::endianness target_endianness, bool is64Bits>
void ELFSymbolTableChunk<target_endianness, is64Bits>::addSymbol(Elf_Sym *sym){
   _symbolTable.push_back(sym);
   this->_size+= sizeof(Elf_Sym) ;
}

/// \brief Add symbols to symbol table
/// We examine each property of atom to infer the various st_* fields in Elf*_Sym
template< support::endianness target_endianness, bool is64Bits>
void ELFSymbolTableChunk<target_endianness, is64Bits>
                  ::addSymbol(const Atom *a, uint16_t shndx) {
 Elf_Sym *symbol = new(_symbolAllocate.Allocate<Elf_Sym>()) Elf_Sym;
 unsigned char b = 0, t = 0;
 symbol->st_name = _stringSection->addString(a->name());
 _symbolToAtom[symbol] = a;
// In relocatable files, st_value holds a section offset for a defined symbol.
// st_value is an offset from the beginning of the section that st_shndx
// identifies. After we assign file offsets we can set this value correctly.
 symbol->st_size = 0;
 symbol->st_shndx = shndx;
 symbol->st_value = 0;
// FIXME: Need to change and account all STV* when visibilities are supported
 symbol->st_other = ELF::STV_DEFAULT;
 if (const DefinedAtom *da = llvm::dyn_cast<const DefinedAtom>(a)){
    symbol->st_size = da->size();
    lld::DefinedAtom::ContentType ct;
    switch (ct = da->contentType()){
    case  DefinedAtom::typeCode:
      t = ELF::STT_FUNC;
      break;
    case  DefinedAtom::typeData:
      t = ELF::STT_OBJECT;
      break;
    case  DefinedAtom::typeZeroFill:
   // In relocatable files, st_value holds alignment constraints for a symbol whose 
   // section index is SHN_COMMON
      if (this->_options.type() == ELF::ET_REL){
        t = ELF::STT_COMMON;
        symbol->st_value = 1 << (da->alignment()).powerOf2;
        symbol->st_shndx = ELF::SHN_COMMON;
      }
      break;
    case DefinedAtom::typeFirstInSection:
      t = ELF::STT_SECTION;
      break;
   // TODO:: How to find STT_FILE symbols?
    default:
      t = ELF::STT_NOTYPE;
    }
 
    if (da->scope() == DefinedAtom::scopeTranslationUnit)  
      b = ELF::STB_LOCAL;
    else if (da->merge() == DefinedAtom::mergeAsWeak)
      b = ELF::STB_WEAK;
    else
      b = ELF::STB_GLOBAL;
  } else if (const AbsoluteAtom *aa = llvm::dyn_cast<const AbsoluteAtom>(a)){
//FIXME: Absolute atoms need more properties to differentiate each other
// based on binding and type of symbol
    t = ELF::STT_OBJECT;

    switch (aa->scope()) {
    case AbsoluteAtom::scopeLinkageUnit:
      symbol->st_other = ELF::STV_HIDDEN;
      b = ELF::STB_LOCAL;
      break;
    case AbsoluteAtom::scopeTranslationUnit:
      b = ELF::STB_LOCAL;
      break;
    case AbsoluteAtom::scopeGlobal:
      b = ELF::STB_GLOBAL;
      break;
    }
    symbol->st_value = aa->value();
 } else {
   symbol->st_value = 0;
   t = ELF::STT_NOTYPE;
   b = ELF::STB_LOCAL;
 }
 symbol->setBindingAndType(b, t);

 _symbolTable.push_back(symbol);
 this->_size += sizeof(Elf_Sym);
}

template<support::endianness target_endianness, bool is64Bits>
void ELFSymbolTableChunk<target_endianness, is64Bits>::setAttributes() {
// sh_info should be one greater than last symbol with STB_LOCAL binding
// we sort the symbol table to keep all local symbols at the beginning
 std::stable_sort(_symbolTable.begin(), _symbolTable.end(), ([]
                  (const Elf_Sym *A, const Elf_Sym *B) -> bool {
                   return (A->getBinding() < B->getBinding());}));
  uint16_t shInfo = 0;
  for (auto i : _symbolTable) {
    if (i->getBinding() != ELF::STB_LOCAL)
      break;
    shInfo++;
  }
  this->_shinfo = shInfo;
// we set the associated string table index in th sh_link member
  this->_link = this->_writer.strtab()->ordinal() - 1;
  this->_align2 = this->_options.pointerWidth();
}

template<support::endianness target_endianness, bool is64Bits>
const char *ELFSymbolTableChunk<target_endianness, is64Bits>::info() {
  return "Symbol Table";
}

template<support::endianness target_endianness, bool is64Bits>
void ELFSymbolTableChunk<target_endianness, is64Bits>::
     write(uint8_t *chunkBuffer) {
  uint64_t chunkOffset = 0;
  for (auto it : _symbolTable) {
    ::memcpy(chunkBuffer + chunkOffset, it, this->_entsize);
    chunkOffset += this->_entsize;
  }
}

//===----------------------------------------------------------------------===//
//  ELFHeaderChunk
//===----------------------------------------------------------------------===//
template<support::endianness target_endianness, bool is64Bits>
ELFHeaderChunk<target_endianness, is64Bits>
              ::ELFHeaderChunk(const WriterOptionsELF &options,
                               const File &File) 
  : Chunk<target_endianness, is64Bits>(Chunk<target_endianness, is64Bits>
                                       ::Kind::Header){
  this->_size = size();
  e_ident(ELF::EI_MAG0, 0x7f);
  e_ident(ELF::EI_MAG1, 'E');
  e_ident(ELF::EI_MAG2, 'L');
  e_ident(ELF::EI_MAG3, 'F');
  e_ident(ELF::EI_CLASS, (options.is64Bit() ? ELF::ELFCLASS64
                                            : ELF::ELFCLASS32));
  e_ident(ELF::EI_DATA, (options.endianness() == llvm::support::big)
                         ? ELF::ELFDATA2MSB
                         : ELF::ELFDATA2LSB);
  e_ident(ELF::EI_VERSION, 1);
  e_ident(ELF::EI_OSABI, ELF::ELFOSABI_NONE);

  e_type(options.type());
  e_machine(options.machine());
  e_version(1);

  e_entry(0ULL);
  e_phoff(0);
  e_shoff(0ULL);
  
  e_flags(0);
  e_ehsize(this->_size);
  e_phentsize(0);
  e_phnum(0);
  e_shentsize(0);
  e_shnum(0);
  e_shstrndx(0);
  this->setGroup(CG_HEADER);
}

template<support::endianness target_endianness, bool is64Bits>
StringRef ELFHeaderChunk<target_endianness, is64Bits>
                        ::segmentName() const {
  return "ELF";
}

template<support::endianness target_endianness, bool is64Bits>
void ELFHeaderChunk<target_endianness, is64Bits>
                   ::write(uint8_t *chunkBuffer) {
  ::memcpy(chunkBuffer, &_eh, size());
}

template<support::endianness target_endianness, bool is64Bits>
const char *ELFHeaderChunk<target_endianness, is64Bits>::info() {
  return "elf_header";
}

//===----------------------------------------------------------------------===//
//  ELFSectionHeaderChunk
//  List of Section Headers:
//[Nr] Name              Type            Addr     Off    Size   ES Flg Lk Inf Al
//[ 0]                   NULL            00000000 000000 000000 00      0   0  0
//[ 1] .text             PROGBITS        00000000 000034 000040 00  AX  0   0  4
//===----------------------------------------------------------------------===//
template<support::endianness target_endianness, bool is64Bits>
ELFSectionHeaderChunk<target_endianness, is64Bits>
                     ::ELFSectionHeaderChunk(const WriterOptionsELF& options,
                                             ELFWriter<target_endianness, 
                                                       is64Bits> &writer) 
  : Chunk<target_endianness, is64Bits>(Chunk<target_endianness, is64Bits>
                                       ::Kind::Header)
  , _options(options)
  , _writer(writer) {
  this->_size = 0;
  this->_align2 = 0;
  // The first element in the list is always NULL
  Elf_Shdr *nullshdr = new (_sectionAllocate.Allocate<Elf_Shdr>()) Elf_Shdr;
  ::memset(nullshdr, 0, sizeof (Elf_Shdr));
  _sectionInfo.push_back(nullshdr);
  this->_size += sizeof (Elf_Shdr);
  this->setGroup(CG_FILE_END);
  }


template<support::endianness target_endianness, bool is64Bits>
void ELFSectionHeaderChunk<target_endianness, is64Bits>::computeSize(){
  this->_size = (this->_writer.sectionChunks().size() + 1) * sizeof(Elf_Shdr);
}

template<support::endianness target_endianness, bool is64Bits>
void ELFSectionHeaderChunk<target_endianness, is64Bits>::fixOffsets(){
  auto it = _sectionInfo.begin();
  auto sections =  _writer.sectionChunks();
  // First section is a NULL section with no sh_offset fix
  (*it)->sh_offset = 0;
  (*it)->sh_addr = 0;
  ++it;
  for (auto &chunk : sections){
    (*it)->sh_offset = chunk->fileOffset();
    (*it)->sh_addr = chunk->address();
    ++it;
  }
}

template<support::endianness target_endianness, bool is64Bits>
void ELFSectionHeaderChunk<target_endianness, is64Bits>::createHeaders(){
  ELFStringSectionChunk<target_endianness, is64Bits> *str = _writer.shstrtab();

 for (const  auto &chunk : _writer.sectionChunks()) {
    Elf_Shdr *shdr  = new (_sectionAllocate.Allocate<Elf_Shdr>()) Elf_Shdr;
    StringRef Name  = chunk->sectionName();
    if (chunk->shStrtableOffset() == 0){
      chunk->setShStrtableOffset(str->addString(Name));
    }
    shdr->sh_name   = chunk->shStrtableOffset();

    shdr->sh_type   = chunk->type();
    shdr->sh_flags  = chunk->flags();
    // TODO: At the time of creation of this section header, we will not have
    // any address and offset info. We  revisit this after assigning the file
    // offsets.
    shdr->sh_offset = chunk->fileOffset();
    shdr->sh_addr   = chunk->address();
    shdr->sh_size   = chunk->size();
    shdr->sh_link = chunk->link() ;
    shdr->sh_info = chunk->shinfo();
    shdr->sh_addralign = chunk->align2();
    shdr->sh_entsize = chunk->entsize();

    _sectionInfo.push_back(shdr);
    this->_size += sizeof (Elf_Shdr);
    _writer.symtab()->setAttributes();
  }
}

template<support::endianness target_endianness, bool is64Bits>
StringRef ELFSectionHeaderChunk<target_endianness, is64Bits>
                               ::segmentName() const {
  return "PT_NULL";
}

template<support::endianness target_endianness, bool is64Bits>
void ELFSectionHeaderChunk<target_endianness, is64Bits>
                          ::write(uint8_t *chunkBuffer) {
  for (const auto si : _sectionInfo) {
    ::memcpy(chunkBuffer, si, sizeof(*si));
    chunkBuffer += sizeof (*si);
  }
}

template<support::endianness target_endianness, bool is64Bits>
uint16_t ELFSectionHeaderChunk<target_endianness, is64Bits>::count() {
  return _sectionInfo.size();
}
template<support::endianness target_endianness, bool is64Bits>
uint16_t ELFSectionHeaderChunk<target_endianness, is64Bits>::size() {
  return sizeof (Elf_Shdr);
}

template<support::endianness target_endianness, bool is64Bits>
const char *ELFSectionHeaderChunk<target_endianness, is64Bits>::info() {
  return "elf_section_header";
}

//===----------------------------------------------------------------------===//
//  ELFProgramHeaderChunk
//===----------------------------------------------------------------------===//
template<support::endianness target_endianness, bool is64Bits>
ELFProgramHeaderChunk<target_endianness, is64Bits>
          ::ELFProgramHeaderChunk(const WriterOptionsELF &options, 
                                  ELFWriter<target_endianness, is64Bits>
                                                 &writer)
  : Chunk<target_endianness, is64Bits>(Chunk<target_endianness, is64Bits>
                                       ::Kind::Header)
  , _options(options)
  , _writer(writer) {
  this->_align2 = 0;
  this->setGroup(CG_HEADER);
  _groupToPF[CG_RWX] = ELF::PF_R | ELF::PF_W | ELF::PF_X;
  _groupToPF[CG_RW]  = ELF::PF_R | ELF::PF_W;
  _groupToPF[CG_RX]  = ELF::PF_R | ELF::PF_X;
  _groupToPF[CG_R]   = ELF::PF_R;
  _groupToPF[CG_W]   = ELF::PF_W;
  _groupToPF[CG_WX]  = ELF::PF_W | ELF::PF_X;
  _groupToPF[CG_X]   = ELF::PF_X;
  }

template<support::endianness target_endianness, bool is64Bits>
void ELFProgramHeaderChunk<target_endianness, is64Bits>::createPHeaders() {

  //TODO Once dynamic linking is supported, implement PHDR segment
  //     Implement mechanism inside this class to correctly find
  //     attributes for sections such as eh_frame, note etc 
  //     when they are supported.
  const uint16_t seg[] = { CG_RWX, CG_RX, CG_R, CG_RW, CG_WX, CG_W, CG_X };
  std::pair<secIterator, secIterator> sectionPair;
  auto sections = _writer.sectionChunks();
  _programHeaders.clear();
  this->_size = 0;
  for (auto group : seg) {
    uint64_t size = 0, mSize = 0;
    sectionPair = std::equal_range(sections.begin(), sections.end(), 
                                   group, ChunkComparator<target_endianness, 
                                                      is64Bits>());
    if (sectionPair.first != sectionPair.second) {
      // FIXME: fix occupiesNoDiskSpace() function in Chunks class
      auto segBegin = sectionPair.first, segEnd = segBegin + 1;
      // Since  this group has a section, atleast this is a part of segment
      size = (*segBegin)->occupiesNoDiskSpace() ? 0 : (*segBegin)->size();
      mSize = (*segBegin)->size();

      for (; segEnd != sectionPair.second; segEnd++) {
      // This means there are more than 1 sections of same permissions
        if ((*segEnd)->fileOffset() - (*segBegin)->fileOffset() - size >
             _options.pageSize()) {
          // we have a case where padding zeros span more than a page
          // we can skip those pages.
          Elf_Phdr *phdr = new(_headerAllocate.Allocate<Elf_Phdr>()) Elf_Phdr;
          phdr->p_type = ELF::PT_LOAD;
          phdr->p_offset = (*segBegin)->fileOffset();
          phdr->p_vaddr =  (*segBegin)->address();
          phdr->p_paddr = phdr->p_vaddr;
          phdr->p_filesz = size;
          // memory size may be more than file size if there are sections
          // that do not occupy space on disk such as .bss
          phdr->p_memsz = mSize;
          phdr->p_flags = _groupToPF[group];
          phdr->p_align = _options.pageSize();
          _programHeaders.push_back(phdr);
          this->_size += sizeof(Elf_Phdr);
          segBegin = segEnd;
          size = (*segBegin)->occupiesNoDiskSpace() ? 0 : (*segBegin)->size();
          mSize = (*segBegin)->size();
        } else {
          size = (*segEnd)->fileOffset() - (*segBegin)->fileOffset() + 
                 ((*segEnd)->occupiesNoDiskSpace() ? 0 : (*segEnd)->size())  ;
          mSize =  (*segEnd)->fileOffset() - (*segBegin)->fileOffset() + 
                   (*segEnd)->size();
        }
      }

      Elf_Phdr *phdr = new(_headerAllocate.Allocate<Elf_Phdr>()) Elf_Phdr;
      phdr->p_type = ELF::PT_LOAD;
      phdr->p_offset = (*segBegin)->fileOffset();
      phdr->p_vaddr = (*segBegin)->address();
      phdr->p_paddr = phdr->p_vaddr;
      phdr->p_filesz = size;
      phdr->p_memsz = mSize;
      phdr->p_flags = _groupToPF[group];
      phdr->p_align = _options.pageSize();
      _programHeaders.push_back(phdr);
      this->_size += sizeof(Elf_Phdr);
    }
  }
}

template<support::endianness target_endianness, bool is64Bits>
uint64_t ELFProgramHeaderChunk<target_endianness, is64Bits>::computeNumber() {
  return _programHeaders.size();
}
template<support::endianness target_endianness, bool is64Bits>
const char *ELFProgramHeaderChunk<target_endianness, is64Bits>::info() {
  return "elf_program_header";
}

template<support::endianness target_endianness, bool is64Bits>
void ELFProgramHeaderChunk<target_endianness, is64Bits>
                          ::write(uint8_t *chunkBuffer) {
  for (const auto si : _programHeaders) {
    ::memcpy(chunkBuffer, si, sizeof(*si));
    chunkBuffer += sizeof (*si);
  }
}
template<support::endianness target_endianness, bool is64Bits>
StringRef ELFProgramHeaderChunk<target_endianness, is64Bits>
                               ::segmentName() const {
  return "PT_NULL";
}
//===----------------------------------------------------------------------===//
//  ELFWriter Class
//===----------------------------------------------------------------------===//
template<support::endianness target_endianness, bool is64Bits>
class ELFWriter : public Writer {
public:
  LLVM_ELF_IMPORT_TYPES(target_endianness, is64Bits)
  typedef object::Elf_Shdr_Impl<target_endianness, is64Bits> Elf_Shdr;
  typedef object::Elf_Sym_Impl<target_endianness, is64Bits> Elf_Sym;
  typedef object::Elf_Phdr<target_endianness, is64Bits> Elf_Phdr;
  ELFWriter(const WriterOptionsELF &options);
  virtual error_code writeFile(const lld::File &File, StringRef path);
  uint64_t addressOfAtom(const Atom *atom);

  std::vector<Chunk<target_endianness, is64Bits>*> chunks() const {
    return _chunks; }

  KindHandler *kindHandler() { return _referenceKindHandler.get(); }
  
  std::vector<SectionChunk<target_endianness, is64Bits>*> sectionChunks() const {
    return _sectionChunks ;
  }
 
  ELFStringSectionChunk<target_endianness, is64Bits> *shstrtab()  const {
    return _shstrtable;
  }
  
  ELFStringSectionChunk<target_endianness, is64Bits> *strtab() const {
    return _strtable;
  }
  ELFSymbolTableChunk<target_endianness, is64Bits> *symtab() const {
    return _symtable;
  }

private:
  void build(const lld::File &file);
  void createChunks(const lld::File &file);
  void buildAtomToAddressMap();
  void assignFileOffsets();
  const WriterOptionsELF &_options;

/// \brief AtomToAddress: Is a mapping from an Atom to the address where
/// it will live in the output file.
  typedef llvm::DenseMap<const Atom*, uint64_t> AtomToAddress;
  typedef typename std::vector<Chunk<target_endianness, is64Bits>*>
                     ::iterator chunkIterator;
  ELFProgramHeaderChunk<target_endianness, is64Bits> *_phdr;
  ELFHeaderChunk<target_endianness, is64Bits> *ehc;
  ELFStringSectionChunk<target_endianness, is64Bits> *_shstrtable ;
  ELFStringSectionChunk<target_endianness, is64Bits> *_strtable ;
  ELFSymbolTableChunk<target_endianness, is64Bits> *_symtable;
  std::unique_ptr<KindHandler> _referenceKindHandler;
  ELFSectionHeaderChunk<target_endianness, is64Bits> *_sectionHeaderChunk;
  AtomToAddress _atomToAddress;
  std::vector<Chunk<target_endianness, is64Bits>*> _chunks;
  const DefinedAtom *_entryAtom;
  std::vector<SectionChunk<target_endianness, is64Bits>*> _sectionChunks;
  std::vector<StockSectionChunk<target_endianness, is64Bits>*> 
                                                      _stockSectionChunks;
  llvm::BumpPtrAllocator _chunkAllocate;
};

//===----------------------------------------------------------------------===//
//  ELFWriter
//===----------------------------------------------------------------------===//
template<support::endianness target_endianness, bool is64Bits>
ELFWriter<target_endianness, is64Bits>
         ::ELFWriter(const WriterOptionsELF &options)
  : _options(options)
  , _referenceKindHandler(KindHandler::makeHandler(_options.machine(),
                                                   target_endianness))
{}

template<support::endianness target_endianness, bool is64Bits>
void ELFWriter<target_endianness, is64Bits>::build(const lld::File &file){
// Create objects for each chunk.
  createChunks(file);
  assignFileOffsets();
  _phdr->createPHeaders();
  _sectionHeaderChunk->createHeaders();
  // Creating program headers changed its size. so we need to re-assign offsets
  assignFileOffsets();
  _sectionHeaderChunk->fixOffsets();
  _phdr->createPHeaders();
  buildAtomToAddressMap();
  _symtable->fixSymbolValue();
  ehc->e_shentsize(_sectionHeaderChunk->size());
  ehc->e_shnum(_sectionHeaderChunk->count());
  // We need to put the index of section string table in ELF header
  // first two chunks are not sections so we subtract 2 to start sections
  // and add 1 since we have a NULL header
  ehc->e_shstrndx(_shstrtable->ordinal() - 1);
  ehc->e_phnum(_phdr->computeNumber());
  ehc->e_phoff(_phdr->fileOffset());
  ehc->e_phentsize(sizeof(Elf_Phdr));
}

template<support::endianness target_endianness, bool is64Bits>
void ELFWriter<target_endianness, is64Bits>
              ::createChunks (const lld::File &file) {
  std::map<StringRef, StockSectionChunk<target_endianness, is64Bits>*> 
             sectionMap;

// Make header chunk
  ehc = new (_chunkAllocate.Allocate
        <ELFHeaderChunk<target_endianness, is64Bits>>())
        ELFHeaderChunk<target_endianness, is64Bits>(_options, file);
  _chunks.push_back(ehc);

  _phdr = new (_chunkAllocate.Allocate
               <ELFProgramHeaderChunk<target_endianness, is64Bits>>())
                ELFProgramHeaderChunk<target_endianness, is64Bits>(_options, 
                                                                   *this);
  _chunks.push_back(_phdr);
        
// We need to create hand crafted sections such as shstrtab strtab hash and
// symtab to put relevant information in ELF structures and then process the
// atoms.

  _shstrtable = new (_chunkAllocate.Allocate
                     <ELFStringSectionChunk<target_endianness, is64Bits>>()) 
                    ELFStringSectionChunk<target_endianness, is64Bits>
                             (_options, *this, ".shstrtab");
  _shstrtable->setShStrtableOffset(_shstrtable->addString(".shstrtab"));
  _sectionChunks.push_back(_shstrtable);
  
  _strtable = new (_chunkAllocate.Allocate
                     <ELFStringSectionChunk<target_endianness, is64Bits>>()) 
                    ELFStringSectionChunk<target_endianness, is64Bits>
                             (_options, *this, ".strtab");
  _strtable->setShStrtableOffset( _shstrtable->addString(".strtab"));
  _sectionChunks.push_back(_strtable);
  
  _symtable = new (_chunkAllocate.Allocate
                     <ELFSymbolTableChunk<target_endianness, is64Bits>>()) 
                    ELFSymbolTableChunk<target_endianness, is64Bits>
                             (_options, *this, ".symtab");
  _symtable->setShStrtableOffset( _shstrtable->addString(".symtab"));
  _sectionChunks.push_back(_symtable);

//TODO: implement .hash section

  for (const DefinedAtom *a : file.defined() ) {
    StringRef sectionName = a->customSectionName();
    if (a->sectionChoice() == 
        DefinedAtom::SectionChoice::sectionBasedOnContent) {
      if (a->contentType() == DefinedAtom::typeZeroFill)
         sectionName = ".bss";
    }
    auto pos = sectionMap.find(sectionName);
    DefinedAtom::ContentType type = a->contentType();
    if (type != DefinedAtom::typeUnknown){
      if (pos == sectionMap.end()) {
       StockSectionChunk<target_endianness, is64Bits>
                  *chunk = new(_chunkAllocate.Allocate
                               <StockSectionChunk<target_endianness, is64Bits>>
                               ())StockSectionChunk<target_endianness, is64Bits>
                                   (sectionName, true, type, _options, *this);

       sectionMap[sectionName] = chunk;
       chunk->appendAtom(a);
       _sectionChunks.push_back(chunk);
       _stockSectionChunks.push_back(chunk);
                            
      } else {
        pos->second->appendAtom(a);
      }
    }
  }

  for (auto chnk : _sectionChunks)
    _chunks.push_back(chnk);

  _sectionHeaderChunk = new (_chunkAllocate.Allocate<ELFSectionHeaderChunk
                               <target_endianness, is64Bits>>())
                              ELFSectionHeaderChunk
                               <target_endianness, is64Bits>(_options, *this); 
  _chunks.push_back(_sectionHeaderChunk);
  // We sort the chunks based on the group they belong to
  std::stable_sort(_chunks.begin(), _chunks.end(), 
                   chunkGroupSort<target_endianness, is64Bits>);
  
  // The CG_RW group also has to be arranged such that all
  // SHT_NOBITS type of sections (.*bss) are at end of this 
  // "partition" of group.
  chunkIterator cI;
  std::pair<chunkIterator, chunkIterator> range;
  range = std::equal_range(_chunks.begin() + 2, _chunks.end() - 1, CG_RW, 
                           ChunkComparator<target_endianness, is64Bits>());
                           
  cI = std::stable_partition(range.first, range.second, 
                             IsBss<target_endianness, is64Bits>);

  // We reassign all the ordinals since its needed when making headers and 
  // populating symbol table.
  uint64_t i = 0;
  for (auto chnk : _chunks) {
    chnk->setOrdinal(i++);
  }

  // We sort the sections as per new ordinal set after group sorting.
  std::stable_sort(_sectionChunks.begin(), _sectionChunks.end(),([]
  (const SectionChunk<target_endianness, is64Bits> *A, 
   const SectionChunk<target_endianness, is64Bits> *B) -> bool {
     return A->ordinal() < B->ordinal();}));

  // Populate symbol table  with correct st_shndx member.
  if (_options.type() == ELF::ET_REL) {
    for (auto chnk : _sectionChunks ) {
       Elf_Sym *sym  = new (_chunkAllocate.Allocate<Elf_Sym>()) Elf_Sym;
       sym->st_name  = 0;
       sym->st_value = 0;
       sym->st_size  = 0;
       sym->st_other = ELF::STV_DEFAULT;
      // first two chunks are not sections hence we subtract 2 but there is a 
      // NULL section in section table so add 1
       sym->st_shndx = chnk->ordinal() - 1 ;
       sym->setBindingAndType(ELF::STB_LOCAL, ELF::STT_SECTION);
       _symtable->addSymbol(sym);
    }
  }
 for (const auto ssc : _stockSectionChunks){
   for (const auto da : ssc->atoms()) {
     _symtable->addSymbol(std::get<0>(da), ssc->ordinal() -1);
   }
 }
 for (const UndefinedAtom *a : file.undefined()) {
   _symtable->addSymbol(a, ELF::SHN_UNDEF);
 }
 
 for (const AbsoluteAtom *a : file.absolute()) {
   _symtable->addSymbol(a, ELF::SHN_ABS);
 }
}

template<support::endianness target_endianness, bool is64Bits>
void ELFWriter<target_endianness, is64Bits>
              ::buildAtomToAddressMap () {

// _atomToAddress is a DenseMap that maps an atom its file address.
// std::get<1>(ai) is the offset from the start of the section to the atom.
  for (auto chunk : _stockSectionChunks){
    for (auto &ai : chunk->atoms() ) {
      _atomToAddress[std::get<0>(ai)] = chunk->address() + std::get<1>(ai);
    }
  }
}

template<support::endianness target_endianness, bool is64Bits>
void ELFWriter<target_endianness, is64Bits>::assignFileOffsets() {
  DEBUG_WITH_TYPE("WriterELF-layout", dbgs() 
                    << "assign file offsets:\n");
  uint64_t offset = 0;
  uint64_t address = _options.type() == ELF::ET_REL ? 0 : 
                                        _options.baseAddress();
  uint16_t chunkGroup;
  auto chunkIt = _chunks.begin();
  // first (two in case of ET_EXEC or ET_DYN) chunks is (are) not  section(s)
  (*chunkIt)->assignFileOffset(offset, address);
  chunkIt++;
  if (_options.type() == ELF::ET_EXEC || 
      _options.type() == ELF::ET_DYN) {
    (*chunkIt)->assignFileOffset(offset, address);
    chunkIt++;
  }
  while (chunkIt != (_chunks.end() - 1) ) {
    (*chunkIt)->assignFileOffset(offset, address);
    if (_options.type() == ELF::ET_EXEC || 
        _options.type() == ELF::ET_DYN) {
      chunkGroup = (*chunkIt)->group();
      chunkIt++;
    // If the chunk group changes we start on new page
      if (chunkGroup != (*chunkIt)->group() && (*chunkIt)->group() != CG_NO_LOAD 
                                            && (*chunkIt)->group() != CG_FILE_END)
        address = address + _options.pageSize();
    } else {
      chunkIt++;
    }
  }
  (*chunkIt)->assignFileOffset(offset, address);
  ehc->e_shoff(_sectionHeaderChunk->fileOffset());
}

template<support::endianness target_endianness, bool is64Bits>
error_code ELFWriter<target_endianness, is64Bits>
                    ::writeFile(const lld::File &file, StringRef path) {
  build(file);

  uint64_t totalSize = _chunks.back()->fileOffset() + _chunks.back()->size();

  OwningPtr<FileOutputBuffer> buffer;
  error_code ec = FileOutputBuffer::create(path,
                                           totalSize, buffer,
                                           FileOutputBuffer::F_executable);
  if (ec)
    return ec;

  for (auto chunk : _chunks) {
    chunk->write(buffer->getBufferStart() + chunk->fileOffset());
  }
  return buffer->commit();
}

template<support::endianness target_endianness, bool is64Bits>
uint64_t ELFWriter<target_endianness, is64Bits>
                    ::addressOfAtom(const Atom *atom) {
  return _atomToAddress[atom];
}
} // namespace elf

Writer *createWriterELF(const WriterOptionsELF &options) {
  if (!options.is64Bit() && options.endianness() == llvm::support::little)
    return new lld::elf::ELFWriter<support::little, false>(options);
  else if (options.is64Bit() && options.endianness() == llvm::support::little)
    return new lld::elf::ELFWriter<support::little, true>(options);
  else if (!options.is64Bit() && options.endianness() == llvm::support::big)
    return new lld::elf::ELFWriter<support::big, false>(options);
  else if (options.is64Bit() && options.endianness() == llvm::support::big)
    return new lld::elf::ELFWriter<support::big, true>(options);

  llvm_unreachable("Invalid Options!");
}
} // namespace lld
