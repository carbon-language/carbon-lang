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

using namespace llvm;
using namespace llvm::object;
namespace lld {
namespace elf {

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

protected:
  Chunk();

  uint64_t _size;
  uint64_t _address;
  uint64_t _fileOffset;
  uint64_t _align2;
};

/// Pair of atom and offset in section.
typedef std::tuple<const DefinedAtom*, uint64_t> AtomInfo;

/// \brief A Section represents a set of Atoms assigned to a specific ELF
///        Section.
template<support::endianness target_endianness, bool is64Bits>
class SectionChunk : public Chunk<target_endianness, is64Bits> {
public:
  SectionChunk(DefinedAtom::ContentType,
               StringRef sectionName,
               const WriterOptionsELF &options,
               ELFWriter<target_endianness, is64Bits> &writer);

  virtual StringRef        segmentName() const;
  virtual bool             occupiesNoDiskSpace();
  virtual void             write(uint8_t *fileBuffer);
  virtual const char      *info();
  StringRef                sectionName();
  uint32_t                 flags() const;
  uint32_t                 type() const;
  uint32_t                 permissions();
  void                     appendAtom(const DefinedAtom*);
  const ArrayRef<AtomInfo> atoms() const;

private:
  StringRef                               _segmentName;
  StringRef                               _sectionName;
  const WriterOptionsELF                 &_options;
  ELFWriter<target_endianness, is64Bits> &_writer;
  uint32_t                                _flags;
  uint32_t                                _type;
  uint32_t                                _permissions;
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

  uint64_t  size() { return sizeof (Elf_Ehdr); }

  virtual StringRef   segmentName() const;
  virtual void        write(uint8_t *fileBuffer);
  virtual const char *info();

private:
  Elf_Ehdr             _eh;
};


/// \brief An ELFSectionHeaderChunk represents the Elf[32/64]_Shdr structure
///        that is placed right after the ELFHeader.
///
/// When this is finished it will need to update the header with the size and
/// number of section headers, e_shentsize, e_shnum.
template<support::endianness target_endianness, bool is64Bits>
class ELFSectionHeaderChunk : public Chunk<target_endianness, is64Bits> {
public:
  LLVM_ELF_IMPORT_TYPES(target_endianness, is64Bits)
  typedef object::Elf_Shdr_Impl<target_endianness, is64Bits> Elf_Shdr;
  ELFSectionHeaderChunk(const WriterOptionsELF &Options,
                        ELFWriter<target_endianness, is64Bits>&);

  virtual StringRef   segmentName() const;
  virtual void        write(uint8_t *filebuffer);
  virtual const char *info();
  void                computeSize(const lld::File &file);
  uint16_t            count();
  uint16_t            size();

  const ArrayRef<Elf_Shdr*> sectionInfo() {
    return _sectionInfo;
  }

  bool is64Bit() { return _options.is64Bit(); }

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
class ELFStringSectionChunk : public Chunk<target_endianness, is64Bits> {
public:
  LLVM_ELF_IMPORT_TYPES(target_endianness, is64Bits)
  ELFStringSectionChunk(const WriterOptionsELF &Options,
                        ELFWriter<target_endianness, is64Bits> &writer,
                        StringRef secName);

  uint64_t addString(StringRef symName);

  virtual StringRef   segmentName() const;
  virtual void        write(uint8_t *filebuffer);
  virtual const char *info();
  StringRef           sectionName();



private:
  StringRef _segName;
  std::vector<StringRef> _StringSection;
  StringRef _sectionName;
  ELFWriter<target_endianness, is64Bits> &_writer;
  const WriterOptionsELF &_options;

};


/// An ELFProgramHeaderChunk represents the Elf[32/64]_Phdr structure near
/// the start of an ELF executable file. Will need to update ELFHeader's
/// e_phentsize/e_phnum when done.
template<support::endianness target_endianness, bool is64Bits>
class ELFProgramHeaderChunk : public Chunk<target_endianness, is64Bits> {
public:
  LLVM_ELF_IMPORT_TYPES(target_endianness, is64Bits)
  ELFProgramHeaderChunk(ELFHeaderChunk<target_endianness, is64Bits>&,
                        const WriterOptionsELF &options,
                        const File &file);


  virtual StringRef   segmentName() const;
  virtual void        write(uint8_t *filebuffer);
  virtual const char *info();

private:
// TODO: Replace this with correct ELF::* type method
//uint32_t              filetype(WriterOptionsELF::OutputKind kind);
};


//===----------------------------------------------------------------------===//
//  Chunk
//===----------------------------------------------------------------------===//

template<support::endianness target_endianness, bool is64Bits>
Chunk<target_endianness, is64Bits>::Chunk()
 : _size(0), _address(0), _fileOffset(0), _align2(0) {
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
  SectionChunk(DefinedAtom::ContentType type,
               StringRef sectionName,
               const WriterOptionsELF &options,
               ELFWriter<target_endianness, is64Bits> &writer)
  :  _options(options)
  , _writer(writer) {
  switch(type) {
  case DefinedAtom::typeCode:
    _segmentName = "PT_LOAD";
    _sectionName = sectionName;
    _flags       = ELF::SHF_ALLOC | ELF::SHF_EXECINSTR;
    _type        = ELF::SHT_PROGBITS;
    break;
  case DefinedAtom::typeData:
    _segmentName = "PT_LOAD";
    _sectionName = sectionName;
    _flags       = ELF::SHF_ALLOC | ELF::SHF_WRITE;
    _type        = ELF::SHT_PROGBITS;
    break;
  case DefinedAtom::typeZeroFill:
    _segmentName = "PT_LOAD";
    _sectionName = sectionName;
    _flags       = ELF::SHF_ALLOC | ELF::SHF_WRITE;
    _type        = ELF::SHT_NOBITS;
    break;
  case DefinedAtom::typeConstant:
    _segmentName = "PT_LOAD";
    _sectionName = sectionName;
    _flags       = ELF::SHF_ALLOC;
    _type        = ELF::SHT_PROGBITS;
    break;
  default:
    llvm_unreachable("Unhandled content type for section!");
  }
}

template<support::endianness target_endianness, bool is64Bits>
bool SectionChunk<target_endianness, is64Bits>::occupiesNoDiskSpace() {
  return false;
}

template<support::endianness target_endianness, bool is64Bits>
StringRef SectionChunk<target_endianness, is64Bits>::segmentName() const {
  return _segmentName;
}

template<support::endianness target_endianness, bool is64Bits>
StringRef SectionChunk<target_endianness, is64Bits>::sectionName() {
  return _sectionName;
}

template<support::endianness target_endianness, bool is64Bits>
uint32_t SectionChunk<target_endianness, is64Bits>::flags() const {
  return _flags;
}

template<support::endianness target_endianness, bool is64Bits>
uint32_t SectionChunk<target_endianness, is64Bits>::type() const {
  return _type;
}

template<support::endianness target_endianness, bool is64Bits>
uint32_t SectionChunk<target_endianness, is64Bits>::permissions() {
  return _permissions;
}

template<support::endianness target_endianness, bool is64Bits>
const char *SectionChunk<target_endianness, is64Bits>::info() {
  return _sectionName.data();
}

template<support::endianness target_endianness, bool is64Bits>
const ArrayRef<AtomInfo> SectionChunk<target_endianness, is64Bits>::
    atoms() const {
  return _atoms;
}


template<support::endianness target_endianness, bool is64Bits>
void SectionChunk<target_endianness, is64Bits>::
    appendAtom(const DefinedAtom *atom) {
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

  // TODO: Check content permissions and figure out what to do with .bss
  if ((perms & DefinedAtom::permR__) == DefinedAtom::permR__)
    this->_permissions |= ELF::SHF_ALLOC;
  if ((perms & DefinedAtom::permRW_) == DefinedAtom::permRW_)
    this->_permissions |= (ELF::SHF_ALLOC | ELF::SHF_WRITE);
  if ((perms & DefinedAtom::permR_X) == DefinedAtom::permR_X)
    this->_permissions |= (ELF::SHF_ALLOC | ELF::SHF_EXECINSTR);
}

template<support::endianness target_endianness, bool is64Bits>
void SectionChunk<target_endianness, is64Bits>::write(uint8_t *chunkBuffer) {
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
         targetAddress = _writer.addressOfAtom(ref->target());

      uint64_t fixupAddress = _writer.addressOfAtom(std::get<0>(ai)) + offset;
      _writer.kindHandler()->applyFixup(ref->kind(), ref->addend(),
                                        &atomContent[offset],
                                        fixupAddress,
                                        targetAddress);
    }
  }
}
//
//===----------------------------------------------------------------------===//
//  ELFStringSectionChunk
//===----------------------------------------------------------------------===//
template<support::endianness target_endianness, bool is64Bits>
ELFStringSectionChunk<target_endianness, is64Bits>::
    ELFStringSectionChunk(const WriterOptionsELF &options,
                          ELFWriter<target_endianness, is64Bits> &writer, 
                          StringRef secName) 
  : _segName("PT_NULL")
  , _sectionName(secName)
  , _writer(writer)
  , _options(options) {
  // First Add a null character. It also occupies 1 byte
  _StringSection.emplace_back("");
  this->_size = 1;
}

template<support::endianness target_endianness, bool is64Bits>
uint64_t ELFStringSectionChunk<target_endianness, is64Bits>::
         addString(StringRef symName) {
  _StringSection.emplace_back(symName);
  
  uint64_t offset = this->_size;
  this->_size += symName.size() + 1;

  return offset;
}

template<support::endianness target_endianness, bool is64Bits>
const char *ELFStringSectionChunk<target_endianness, is64Bits>::info() {
  return _sectionName.data();
}

template<support::endianness target_endianness, bool is64Bits>
StringRef ELFStringSectionChunk<target_endianness, is64Bits>::sectionName() {
  return _sectionName ;
}

template<support::endianness target_endianness, bool is64Bits>
StringRef ELFStringSectionChunk<target_endianness, is64Bits>::
          segmentName() const {
  return _segName;
}

// We need to unwrap the _StringSection and then make one large memory 
// chunk of null terminated strings
template<support::endianness target_endianness, bool is64Bits>
void ELFStringSectionChunk<target_endianness, is64Bits>::
     write(uint8_t *chunkBuffer) {
  uint64_t chunkOffset = 0;
 
  for (auto it : _StringSection) {
    ::memcpy(chunkBuffer + chunkOffset, it.data(), it.size());
    chunkOffset += it.size();
    ::memcpy(chunkBuffer + chunkOffset, "", 1);
    chunkOffset += 1;
  }
}

//===----------------------------------------------------------------------===//
//  ELFHeaderChunk
//===----------------------------------------------------------------------===//
template<support::endianness target_endianness, bool is64Bits>
ELFHeaderChunk<target_endianness, is64Bits>
              ::ELFHeaderChunk(const WriterOptionsELF &options,
                               const File &File) {
  this->_size = size();
  e_ident(ELF::EI_MAG0, 0x7f);
  e_ident(ELF::EI_MAG1, 'E');
  e_ident(ELF::EI_MAG2, 'L');
  e_ident(ELF::EI_MAG3, 'F');
  e_ident(ELF::EI_CLASS, (options.is64Bit() ? ELF::ELFCLASS64
                                            : ELF::ELFCLASS32));
  e_ident(ELF::EI_DATA, options.endianness());
  e_ident(ELF::EI_VERSION, 1);
  e_ident(ELF::EI_OSABI, ELF::ELFOSABI_NONE);

  e_type(options.type());
  e_machine(options.machine());
  e_version(1);

  e_entry(0ULL);
  e_phoff(this->_size);
  e_shoff(0ULL);
  
  e_flags(0);
  e_ehsize(this->_size);
  e_phentsize(0);
  e_phnum(0);
  e_shentsize(0);
  e_shnum(0);
  e_shstrndx(0);
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
  : _options(options)
  , _writer(writer) {

  this->_size = 0;
  this->_align2 = 0;
  // The first element in the list is always NULL
  Elf_Shdr *nullshdr = new (_sectionAllocate.Allocate<Elf_Shdr>()) Elf_Shdr;
  ::memset(nullshdr, 0, sizeof (Elf_Shdr));
  _sectionInfo.push_back(nullshdr);

  this->_size += sizeof (Elf_Shdr);

  ELFStringSectionChunk<target_endianness, is64Bits> *str = _writer.shstrtab();

  for (const auto &chunk : _writer.sectionChunks()) {
    Elf_Shdr *shdr  = new (_sectionAllocate.Allocate<Elf_Shdr>()) Elf_Shdr;
    StringRef Name  = chunk->sectionName();
    uint64_t offset = str->addString(Name);
    shdr->sh_name   = offset;
    shdr->sh_type   = chunk->type();
    shdr->sh_flags  = chunk->flags();
    // TODO: At the time of creation of this section header, we will not have
    // any address and offset info. We  revisit this after assigning the file
    // offsets.
    shdr->sh_offset = chunk->fileOffset();
    shdr->sh_addr   = chunk->address();
    shdr->sh_size   = chunk->size();
// The next two fields have special meanings:
// sh_type           sh_link                             sh_info
// SHT_DYNAMIC  The section header index of the string
//                table used by entries in the section.   0
// SHT_HASH     The section header index of the symbol
//                table to which the hash table applies.  0
// SHT_REL
// SHT_RELA     The section header index of the
//                associated symbol table.                The section header
//                                                        index of the section
//                                                        to which the
//                                                        relocation applies.
// SHT_SYMTAB
// SHT_DYNSYM   The section header index of the
//                associated string table.                One greater than the
//                                                        symbol table index of
//                                                        the last local symbol
//                                                        (binding STB_LOCAL).
// SHT_GROUP    The section header index of the
//              associated symbol table.                  The symbol table
//                                                        index of an entry in
//                                                        the associated symbol
//                                                        table. The name of
//                                                        the specified symbol
//                                                        table entry provides
//                                                        a signature for the
//                                                        section group.
// SHT_SYMTAB_SHNDX The section header index of
//                  the associated symbol table section.  0
// None of these chunks are of the above mentioned type, so we short them.
    shdr->sh_link = 0;
    shdr->sh_info = 0;
    shdr->sh_addralign = chunk->align2();
    // Not a special section with fixed entries
    shdr->sh_entsize = 0;

    _sectionInfo.push_back(shdr);
    this->_size += sizeof (Elf_Shdr);
  }

  // Now I add in the section string table. For some reason This seems to be 
  // preferred location of string sections in contemporary
  // (ones that must not be named) linker(s).
  Elf_Shdr *shdr = new (_sectionAllocate.Allocate<Elf_Shdr>()) Elf_Shdr;
  // I push the name of the string table into the string table as soon as
  // it is created.
  shdr->sh_name   = 1;
  shdr->sh_type   = ELF::SHT_STRTAB;
  shdr->sh_flags  = 0;
  // NOTE: Refer to above note when assigning st_addr for other sections.
  shdr->sh_addr   = str->address();
  shdr->sh_offset = str->fileOffset();
  shdr->sh_size   = str->size();
  shdr->sh_link   = 0;
  shdr->sh_info   = 0;
  // This section is not a loadable section, hence we do not care about
  // alignment.
  shdr->sh_addralign = 1;
  _sectionInfo.push_back(shdr);
  this->_size += sizeof (Elf_Shdr);
}

template<support::endianness target_endianness, bool is64Bits>
StringRef ELFSectionHeaderChunk<target_endianness, is64Bits>
                               ::segmentName() const {
  return "SHDR";
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
// TODO: Implement the methods

//===----------------------------------------------------------------------===//
//  ELFWriter Class
//===----------------------------------------------------------------------===//
template<support::endianness target_endianness, bool is64Bits>
class ELFWriter : public Writer {
public:
  LLVM_ELF_IMPORT_TYPES(target_endianness, is64Bits)
  typedef object::Elf_Shdr_Impl<target_endianness, is64Bits> Elf_Shdr;
  ELFWriter(const WriterOptionsELF &options);
  virtual error_code writeFile(const lld::File &File, StringRef path);
  uint64_t addressOfAtom(const Atom *atom);
  ArrayRef<Chunk<target_endianness, is64Bits>*> chunks() { return _chunks; }
  KindHandler *kindHandler() { return _referenceKindHandler.get(); }
  
  std::vector<SectionChunk<target_endianness, is64Bits>*> sectionChunks() {
    return _sectionChunks ;
  }
  
  ELFStringSectionChunk<target_endianness, is64Bits> *shstrtab() {
    return _shstrtable;
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

  ELFStringSectionChunk<target_endianness, is64Bits> *_shstrtable ;
  std::unique_ptr<KindHandler> _referenceKindHandler;
  ELFSectionHeaderChunk<target_endianness, is64Bits> *_sectionHeaderChunk;
  AtomToAddress _atomToAddress;
  std::vector<Chunk<target_endianness, is64Bits>*> _chunks;
  const DefinedAtom *_entryAtom;
  std::vector<SectionChunk<target_endianness, is64Bits>*> _sectionChunks;
  llvm::BumpPtrAllocator _chunkAllocate;
};

//===----------------------------------------------------------------------===//
//  ELFWriter
//===----------------------------------------------------------------------===//
template<support::endianness target_endianness, bool is64Bits>
ELFWriter<target_endianness, is64Bits>
         ::ELFWriter(const WriterOptionsELF &options)
  : _options(options)
  , _referenceKindHandler(KindHandler::makeHandler(_options.machine()))
{}

template<support::endianness target_endianness, bool is64Bits>
void ELFWriter<target_endianness, is64Bits>::build(const lld::File &file){
  // Create objects for each chunk.
  createChunks(file);
  assignFileOffsets();
  buildAtomToAddressMap();
}

template<support::endianness target_endianness, bool is64Bits>
void ELFWriter<target_endianness, is64Bits>
              ::createChunks (const lld::File &file) {
  std::map<StringRef, SectionChunk<target_endianness, is64Bits>*> sectionMap;

  // We need to create hand crafted sections such as shstrtab strtab hash and
  // symtab to put relevant information in ELF structures and then process the
  // atoms.

  _shstrtable = new (_chunkAllocate.Allocate
                     <ELFStringSectionChunk<target_endianness, is64Bits>>()) 
                    ELFStringSectionChunk<target_endianness, is64Bits>
                             (_options, *this, ".shstrtab");
  _shstrtable->addString(".shstrtab");

  //we also need to process undefined atoms
  for (const DefinedAtom *a : file.defined() ) {
    // TODO: Add sectionChoice.
    // assert( atom->sectionChoice() == DefinedAtom::sectionBasedOnContent );
    StringRef sectionName = a->customSectionName();
    auto pos = sectionMap.find(sectionName);
    DefinedAtom::ContentType type = a->contentType();
    if (pos == sectionMap.end()) {
      if (type != DefinedAtom::typeUnknown){
    	  SectionChunk<target_endianness, is64Bits>
                  *chunk = new (_chunkAllocate.Allocate
                                <SectionChunk<target_endianness, is64Bits>>())
                                SectionChunk<target_endianness, is64Bits>
                                   (type, sectionName, _options, *this);

         sectionMap[sectionName] = chunk;
         chunk->appendAtom(a);
         _sectionChunks.push_back(chunk);
      }
    } else {
      pos->second->appendAtom(a);
    }
  }

  //put in the Undefined atoms as well
  // Make header chunk
  ELFHeaderChunk<target_endianness, is64Bits> *ehc = 
    new (_chunkAllocate.Allocate
        <ELFHeaderChunk<target_endianness, is64Bits>>())
        ELFHeaderChunk<target_endianness, is64Bits>(_options, file);

  _sectionHeaderChunk = new (_chunkAllocate.Allocate<ELFSectionHeaderChunk
                               <target_endianness, is64Bits>>())
                              ELFSectionHeaderChunk
                               <target_endianness, is64Bits>(_options, *this); 

  ehc->e_shoff(ehc->size());
  ehc->e_shentsize(_sectionHeaderChunk->size());
  ehc->e_shnum(_sectionHeaderChunk->count());
 
   // I am pushing string section after all sections are in.
   // Hence the index will be total number of non-custom sections we have

  ehc->e_shstrndx(_sectionChunks.size() + 1);
  _chunks.push_back(ehc);
  _chunks.push_back(_sectionHeaderChunk);
  // We have ELF header, section header. Now push rest of sections
  for (auto chnk : _sectionChunks)
    _chunks.push_back(chnk);
  _chunks.push_back(_shstrtable);
}

template<support::endianness target_endianness, bool is64Bits>
void ELFWriter<target_endianness, is64Bits>
              ::buildAtomToAddressMap () {

// _atomToAddress is a DenseMap that maps an atom its file address.
// std::get<1>(ai) is the offset from the start of the section to the atom.
  for (auto &chunk : _sectionChunks){
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
  uint64_t address = 0;
  for (auto chunk : _chunks) {

    chunk->assignFileOffset(offset, address);
  }
  //TODO: We need to fix all file offsets inside various ELF section headers
  std::vector<Elf_Shdr*> secInfo = _sectionHeaderChunk->sectionInfo();
  typename std::vector<Elf_Shdr*>::iterator it = secInfo.begin();
  // First section is a NULL section with no sh_offset fix
  (*it)->sh_offset = 0;
  (*it)->sh_addr = 0;
  ++it;
  for (auto &chunk : _sectionChunks){
    (*it)->sh_offset = chunk->fileOffset();
    (*it)->sh_addr = chunk->address();
    ++it;
  }
  // We  have taken care of  all the stock sections. We need to deal with 
  // custom sections
  // They are section string table, string table and symbol table
  (*it)->sh_offset = _shstrtable->fileOffset();
  (*it)->sh_addr = _shstrtable->address();
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
