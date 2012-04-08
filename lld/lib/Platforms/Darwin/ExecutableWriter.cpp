//===- Platforms/Darwin/ExecutableWriter.cpp ------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ExecutableWriter.h"
#include "MachOFormat.hpp"
#include "DarwinReferenceKinds.h"
#include "DarwinPlatform.h"

#include <vector>
#include <map>

#include <string.h>

#include "llvm/Support/Debug.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

#include "lld/Core/DefinedAtom.h"
#include "lld/Core/SharedLibraryAtom.h"
#include "lld/Core/File.h"
#include "lld/Core/Reference.h"


namespace lld {
namespace darwin {

// 
// A mach-o file consists of some meta data (header and load commands), 
// then atom content (e.g. function instructions), then more meta data
// (symbol table, etc).  Before you can write a mach-o file, you need to
// compute what will be the file offsets and "addresses" of various things
// in the file.  
//
// The design here is to break up what will be the mach-o file into chunks.  
// Each Chunk has an object to manage its size and content.  There is a
// chunk for the mach_header, one for the load commands, and one for each
// part of the LINKEDIT segment.  There is also one chunk for each traditional
// mach-o section.  The MachOWriter manages the list of chunks.  And
// asks each to determine its size in the correct order.  Many chunks
// cannot be sized until other chunks are sized (e.g. the dyld info
// in the LINKEDIT cannot be sized until all atoms have been assigned 
// addresses).
//
// Once all chunks have a size, the MachOWriter iterates through them and
// asks each to write out their content.   
//



//
// A Chunk is an abstrace contiguous range of a generated 
// mach-o executable file.
//
class Chunk {
public:
  virtual StringRef   segmentName() const = 0;
  virtual bool        occupiesNoDiskSpace();
  virtual void        write(raw_ostream &out) = 0;
  static void         writeZeros(uint64_t amount, raw_ostream &out);
  void                assignFileOffset(uint64_t &curOff, uint64_t &curAddr);
  virtual const char* info() = 0;
  uint64_t            size() const;
  uint64_t            address() const;
  uint64_t            fileOffset() const;
  uint64_t            align2() const;
  static uint64_t     alignTo(uint64_t value, uint8_t align2);
  
protected:
                      Chunk();
  
  uint64_t            _size;
  uint64_t            _address;
  uint64_t            _fileOffset;
  uint32_t            _align2;
};



//
// A SectionChunk represents a set of Atoms assigned to a specific 
// mach-o section (which is a subrange of a mach-o segment).  
// For example, there is one SectionChunk for the __TEXT,__text section.
//
class SectionChunk : public Chunk {
public:
  static SectionChunk*  make(DefinedAtom::ContentType, 
                             DarwinPlatform &platform,
                             class MachOWriter &writer);
  virtual StringRef     segmentName() const;
  virtual bool          occupiesNoDiskSpace();
  virtual void          write(raw_ostream &out);
  virtual const char*   info();
  StringRef             sectionName();
  uint32_t              flags() const;
  uint32_t              permissions();
  void                  appendAtom(const DefinedAtom*);

  struct AtomInfo {
    const DefinedAtom  *atom;
    uint64_t            offsetInSection;
  };

  const std::vector<AtomInfo>& atoms() const;

private:
                SectionChunk(StringRef seg, 
                             StringRef sect, 
                             uint32_t flags, 
                             DarwinPlatform &platform,
                             class MachOWriter &writer);

  StringRef                _segmentName;
  StringRef                _sectionName;
  DarwinPlatform          &_platform;
  class MachOWriter       &_writer;
  uint32_t                 _flags;
  uint32_t                 _permissions;
  std::vector<AtomInfo>    _atoms;
};



//
// A MachHeaderChunk represents the mach_header struct at the start 
// of a mach-o executable file.
//
class MachHeaderChunk : public Chunk {
public:
                MachHeaderChunk(DarwinPlatform &plat, const File &file);
  virtual StringRef     segmentName() const;
  virtual void          write(raw_ostream &out);
  virtual const char*   info(); 
  void                  recordLoadCommand(load_command*);
  uint64_t              loadCommandsSize();

private:
  mach_header               _mh;
};



//
// A LoadCommandsChunk represents the variable length list of
// of load commands in a mach-o executable file right after the
// mach_header.
//
class LoadCommandsChunk : public Chunk {
public:
                      LoadCommandsChunk(MachHeaderChunk&, 
                                        DarwinPlatform&,
                                        class MachOWriter&);
  virtual StringRef   segmentName() const;
  virtual void        write(raw_ostream &out);
  virtual const char* info(); 
  void                computeSize(const lld::File &file);
  void                addSection(SectionChunk*);
  void                updateLoadCommandContent(const lld::File &file);

private:
  friend class LoadCommandPaddingChunk;

  void                addLoadCommand(load_command* lc);
  void                setMachOSection(SectionChunk *chunk, 
                                      segment_command_64 *seg, uint32_t index);
  uint32_t            permissionsFromSections(
                                  const SmallVector<SectionChunk*,16> &);

  struct ChunkSegInfo {
    SectionChunk*         chunk;
    segment_command_64*   segment;
    section_64*           section;
  };

  MachHeaderChunk             &_mh;
  DarwinPlatform              &_platform;
  class MachOWriter           &_writer;
  segment_command_64          *_linkEditSegment;
  symtab_command              *_symbolTableLoadCommand;
  entry_point_command         *_entryPointLoadCommand;
  dyld_info_command           *_dyldInfoLoadCommand;
  std::vector<load_command*>   _loadCmds;
  std::vector<ChunkSegInfo>    _sectionInfo;
};



//
// A LoadCommandPaddingChunk represents the padding space between the last 
// load commmand and the first section (usually __text) in the __TEXT
// segment.
//
class LoadCommandPaddingChunk : public Chunk {
public:
                      LoadCommandPaddingChunk(LoadCommandsChunk&);
  virtual StringRef   segmentName() const;
  virtual void        write(raw_ostream &out);
  virtual const char* info(); 
  void                computeSize();
private:
  LoadCommandsChunk&  _loadCommandsChunk;
};



//
// LinkEditChunk is the base class for all chunks in the
// __LINKEDIT segment at the end of a mach-o executable. 
//
class LinkEditChunk : public Chunk {
public:
                      LinkEditChunk();
  virtual StringRef   segmentName() const;
  virtual void        computeSize(const lld::File &file,
                                      const std::vector<SectionChunk*>&) = 0;
};



//
// A DyldInfoChunk represents the bytes for any of the dyld info areas
// in the __LINKEDIT segment at the end of a mach-o executable. 
//
class DyldInfoChunk : public LinkEditChunk {
public:
                      DyldInfoChunk(class MachOWriter &);
  virtual void        write(raw_ostream &out);

protected:
  void                append_byte(uint8_t);
  void                append_uleb128(uint64_t);
  void                append_string(StringRef);

  class MachOWriter      &_writer;
  std::vector<uint8_t>    _bytes;
};



//
// A BindingInfoChunk represents the bytes containing binding info
// in the __LINKEDIT segment at the end of a mach-o executable. 
//
class BindingInfoChunk : public DyldInfoChunk {
public:
                      BindingInfoChunk(class MachOWriter &);
  virtual void        computeSize(const lld::File &file,
                                      const std::vector<SectionChunk*>&);
  virtual const char* info(); 
};



//
// A LazyBindingInfoChunk represents the bytes containing lazy binding info
// in the __LINKEDIT segment at the end of a mach-o executable. 
//
class LazyBindingInfoChunk : public DyldInfoChunk {
public:
                      LazyBindingInfoChunk(class MachOWriter &);
  virtual void        computeSize(const lld::File &file,
                                      const std::vector<SectionChunk*>&);
  virtual const char* info(); 
private:
  void                 updateHelper(const DefinedAtom *, uint32_t );
};
  

//
// A SymbolTableChunk represents the array of nlist structs in the
// __LINKEDIT segment at the end of a mach-o executable. 
//
class SymbolTableChunk : public LinkEditChunk {
public:
                      SymbolTableChunk(class SymbolStringsChunk&);
  virtual void        write(raw_ostream &out);
  virtual void        computeSize(const lld::File &file,
                                      const std::vector<SectionChunk*>&);
  virtual const char* info(); 
  uint32_t            count();

private:
  uint8_t             nType(const DefinedAtom*);

  SymbolStringsChunk       &_stringsChunk;
  std::vector<nlist_64>     _globalDefinedsymbols;
  std::vector<nlist_64>     _localDefinedsymbols;
  std::vector<nlist_64>     _undefinedsymbols;
};


//
// A SymbolStringsChunk represents the strings pointed to
// by nlist structs in the __LINKEDIT segment at the end 
// of a mach-o executable. 
//
class SymbolStringsChunk : public LinkEditChunk {
public:
                      SymbolStringsChunk();
  virtual void        write(raw_ostream &out);
  virtual void        computeSize(const lld::File &file,
                                      const std::vector<SectionChunk*>&);
  virtual const char* info(); 
  uint32_t            stringIndex(StringRef);

private:
  std::vector<char>         _strings;
};


//
// A MachOWriter manages all the Chunks that comprise a mach-o executable.
//
class MachOWriter {
public:
              MachOWriter(DarwinPlatform &platform);
  void        build(const lld::File &file);
  void        write(raw_ostream &out);

  uint64_t    addressOfAtom(const Atom *atom);
  void        zeroFill(int64_t amount, raw_ostream &out);
  void        findSegment(StringRef segmentName, uint32_t *segIndex,
                                uint64_t *segStartAddr, uint64_t *segEndAddr);

  const std::vector<Chunk*> chunks() { return _chunks; }

private:
  friend LoadCommandsChunk;
  friend LazyBindingInfoChunk;

  void        createChunks(const lld::File &file);
  void        buildAtomToAddressMap();
  void        assignFileOffsets();
  void        addLinkEditChunk(LinkEditChunk *chunk);
  void        buildLinkEdit(const lld::File &file);
  void        assignLinkEditFileOffsets();
  void        dump();


  typedef llvm::DenseMap<const Atom*, uint64_t> AtomToAddress;
  
  DarwinPlatform             &_platform;
  LoadCommandsChunk          *_loadCommandsChunk;
  LoadCommandPaddingChunk    *_paddingChunk;
  AtomToAddress               _atomToAddress;
  std::vector<Chunk*>         _chunks;
  std::vector<SectionChunk*>  _sectionChunks;
  std::vector<LinkEditChunk*> _linkEditChunks;
  BindingInfoChunk           *_bindingInfo;
  LazyBindingInfoChunk       *_lazyBindingInfo;
  SymbolTableChunk           *_symbolTableChunk;
  SymbolStringsChunk         *_stringsChunk;
  uint64_t                    _linkEditStartOffset;
  uint64_t                    _linkEditStartAddress;
};



//===----------------------------------------------------------------------===//
//  Chunk
//===----------------------------------------------------------------------===//

Chunk::Chunk() 
 : _size(0), _address(0), _fileOffset(0), _align2(0) {
}

bool Chunk::occupiesNoDiskSpace() { 
  return false; 
}

uint64_t Chunk::size() const {
  return _size;
}

uint64_t Chunk::align2() const {
  return _align2;
}

uint64_t Chunk::address() const {
  return _address;
}

uint64_t Chunk::fileOffset() const {
  return _fileOffset;
}


void Chunk::writeZeros(uint64_t amount, raw_ostream &out) {
  for( int i=amount; i > 0; --i)
    out.write('\0');
}

uint64_t Chunk::alignTo(uint64_t value, uint8_t align2) {
  uint64_t align = 1 << align2;
  return ( (value + (align-1)) & (-align) );
}

void Chunk::assignFileOffset(uint64_t &curOffset, uint64_t &curAddress) {
  if ( this->occupiesNoDiskSpace() ) {
    // FileOffset does not change, but address space does change.
    uint64_t alignedAddress = alignTo(curAddress, _align2);
   _address = alignedAddress;
   curAddress = alignedAddress + _size;
  }
  else {
    // FileOffset and address both move by _size amount after alignment.
    uint64_t alignPadding = alignTo(curAddress, _align2) - curAddress;
    _fileOffset = curOffset + alignPadding;
    _address = curAddress + alignPadding;
    curOffset = _fileOffset + _size;
    curAddress = _address + _size;
  }
  DEBUG(llvm::dbgs() << "   fileOffset=0x";
        llvm::dbgs().write_hex(_fileOffset);
        llvm::dbgs() << " address=0x";
        llvm::dbgs().write_hex(_address);
        llvm::dbgs() << " info=" << this->info() << "\n");
} 



//===----------------------------------------------------------------------===//
//  SectionChunk
//===----------------------------------------------------------------------===//

SectionChunk::SectionChunk(StringRef seg, StringRef sect, 
                           uint32_t flags, DarwinPlatform &platform,
                                                MachOWriter &writer)
 : _segmentName(seg), _sectionName(sect), _platform(platform), 
   _writer(writer), _flags(flags), _permissions(0) {
  
}

SectionChunk* SectionChunk::make(DefinedAtom::ContentType type, 
                                 DarwinPlatform &platform,
                                 MachOWriter &writer) {
  switch ( type ) {
    case DefinedAtom::typeCode:
      return new SectionChunk("__TEXT", "__text", 
                              S_REGULAR | S_ATTR_PURE_INSTRUCTIONS,
                              platform, writer);
      break;
    case DefinedAtom::typeCString:
       return new SectionChunk("__TEXT", "__cstring", 
                               S_CSTRING_LITERALS,
                              platform, writer);
       break;
    case DefinedAtom::typeStub:
      return new SectionChunk("__TEXT", "__stubs", 
                              S_SYMBOL_STUBS | S_ATTR_PURE_INSTRUCTIONS,
                              platform, writer);
      break;
    case DefinedAtom::typeStubHelper:
      return new SectionChunk("__TEXT", "__stub_helper", 
                              S_REGULAR | S_ATTR_PURE_INSTRUCTIONS,
                              platform, writer);
      break;
    case DefinedAtom::typeLazyPointer:
      return new SectionChunk("__DATA", "__la_symbol_ptr", 
                              S_LAZY_SYMBOL_POINTERS,
                              platform, writer);
      break;
    case DefinedAtom::typeGOT:
      return new SectionChunk("__DATA", "__got", 
                              S_NON_LAZY_SYMBOL_POINTERS,
                              platform, writer);
      break;
    default:
      assert(0 && "TO DO: add support for more sections");
      break;
  }
  return nullptr;
}

bool SectionChunk::occupiesNoDiskSpace() {
  return ( (_flags & SECTION_TYPE) == S_ZEROFILL );
}

StringRef SectionChunk::segmentName() const {
  return _segmentName;
}

StringRef SectionChunk::sectionName() {
  return _sectionName;
}

uint32_t SectionChunk::flags() const {
  return _flags;
}

uint32_t SectionChunk::permissions() {
  return _permissions;
}

const char* SectionChunk::info() {
  return _sectionName.data();
}

const std::vector<SectionChunk::AtomInfo>& SectionChunk::atoms() const {
  return _atoms;
}

void SectionChunk::appendAtom(const DefinedAtom *atom) {
  // Figure out offset for atom in this section given alignment constraints.
  uint64_t offset = _size;
  DefinedAtom::Alignment atomAlign = atom->alignment();
  uint64_t align2 = 1 << atomAlign.powerOf2;
  uint64_t requiredModulus = atomAlign.modulus;
  uint64_t currentModulus = (offset % align2);
  if ( currentModulus != requiredModulus ) {
    if ( requiredModulus > currentModulus )
      offset += requiredModulus-currentModulus;
    else
      offset += align2+requiredModulus-currentModulus;
  }
  // Record max alignment of any atom in this section.
  if ( align2 > _align2 )
    _align2 = align2;
  // Assign atom to this section with this offset.
  _atoms.push_back({atom, offset});
  // Update section size to include this atom.
  _size = offset + atom->size();
  // Update permissions
  DefinedAtom::ContentPermissions perms = atom->permissions();
  if ( (perms & DefinedAtom::permR__) == DefinedAtom::permR__ )
    _permissions |= VM_PROT_READ;
  if ( (perms & DefinedAtom::permRW_) == DefinedAtom::permRW_ )
    _permissions |= VM_PROT_WRITE;
  if ( (perms & DefinedAtom::permR_X) == DefinedAtom::permR_X )
    _permissions |= VM_PROT_EXECUTE;
}


void SectionChunk::write(raw_ostream &out) {
  assert( out.tell() == _fileOffset);
  SmallVector<uint8_t, 1024> buffer;
  // Each section's content is just its atoms' content.
  for (const AtomInfo &atomInfo : _atoms ) {
    uint64_t atomFileOffset = _fileOffset + atomInfo.offsetInSection;
    if ( atomFileOffset != out.tell() ) {
      // Need to add alignment padding before this atom starts.
      assert(atomFileOffset > out.tell());
      this->writeZeros(atomFileOffset - out.tell(), out);
    }
    // Copy raw content of atom.
    ArrayRef<uint8_t> content = atomInfo.atom->rawContent();
    buffer.resize(content.size());
    ::memcpy(buffer.data(), content.data(), content.size());
    for (const Reference *ref : *atomInfo.atom) {
      uint32_t offset = ref->offsetInAtom();
      uint64_t targetAddress = 0;
      if ( ref->target() != nullptr )
        targetAddress = _writer.addressOfAtom(ref->target());
      uint64_t fixupAddress = _writer.addressOfAtom(atomInfo.atom) + offset;
      _platform.applyFixup(ref->kind(), ref->addend(), &buffer[offset],
                                              fixupAddress, targetAddress);
    }
    for( uint8_t byte : buffer) {
      out.write(byte);
    }
  }
}

//===----------------------------------------------------------------------===//
//  MachHeaderChunk
//===----------------------------------------------------------------------===//

MachHeaderChunk::MachHeaderChunk(DarwinPlatform &platform, const File &file) {
  // Let platform convert file info to mach-o cpu type and subtype.
  platform.initializeMachHeader(file, _mh);
  _size = _mh.size();
}


StringRef MachHeaderChunk::segmentName() const {
  return StringRef("__TEXT");
}

void MachHeaderChunk::write(raw_ostream &out) {
  assert( out.tell() == _fileOffset);
  _mh.write(out);
}

const char* MachHeaderChunk::info() {
  return "mach_header";
}

void MachHeaderChunk::recordLoadCommand(load_command* lc) {
  _mh.recordLoadCommand(lc);
}

uint64_t MachHeaderChunk::loadCommandsSize() {
  return _mh.sizeofcmds;
}



//===----------------------------------------------------------------------===//
//  LoadCommandsChunk
//===----------------------------------------------------------------------===//

LoadCommandsChunk::LoadCommandsChunk(MachHeaderChunk &mh, 
                                     DarwinPlatform& platform, 
                                     MachOWriter& writer)
 : _mh(mh), _platform(platform), _writer(writer),
   _linkEditSegment(nullptr), _symbolTableLoadCommand(nullptr),
   _entryPointLoadCommand(nullptr), _dyldInfoLoadCommand(nullptr) {
}


StringRef LoadCommandsChunk::segmentName() const {
  return StringRef("__TEXT");
}

void LoadCommandsChunk::write(raw_ostream &out) {
  assert( out.tell() == _fileOffset);
  for ( load_command* lc : _loadCmds ) {
    lc->write(out);
  }
}

const char* LoadCommandsChunk::info() {
  return "load commands";
}

void LoadCommandsChunk::setMachOSection(SectionChunk *chunk, 
                                    segment_command_64 *seg, uint32_t index) {
  for (ChunkSegInfo &entry : _sectionInfo) {
    if ( entry.chunk == chunk ) {
      entry.section = &(seg->sections[index]);
      entry.segment = seg;
      return;
    }
  }
  assert(0 && "setMachOSection() chunk not found");
}

uint32_t LoadCommandsChunk::permissionsFromSections(
                        const SmallVector<SectionChunk*,16> &sections) {
  uint32_t result = 0;
  for (SectionChunk *chunk : sections) {
    result |= chunk->permissions();
  }
  return result;
}

void LoadCommandsChunk::computeSize(const lld::File &file) {
  // Main executables have a __PAGEZERO segment.
  uint64_t pageZeroSize = _platform.pageZeroSize();
  if ( pageZeroSize != 0 ) {
    segment_command_64* pzSegCmd = segment_command_64::make(0);
    strcpy(pzSegCmd->segname, "__PAGEZERO");
    pzSegCmd->vmaddr   = 0;
    pzSegCmd->vmsize   = pageZeroSize;
    pzSegCmd->fileoff  = 0;
    pzSegCmd->filesize = 0;
    pzSegCmd->maxprot  = 0;
    pzSegCmd->initprot = 0;
    pzSegCmd->nsects   = 0;
    pzSegCmd->flags    = 0;
    this->addLoadCommand(pzSegCmd);
  }
  // Add other segment load commands
  StringRef lastSegName = StringRef("__TEXT");
  SmallVector<SectionChunk*,16> sections;
  for (ChunkSegInfo &entry : _sectionInfo) {
    StringRef entryName = entry.chunk->segmentName();
    if ( !lastSegName.equals(entryName) ) {
      // Start of new segment, so create load command for all previous sections.
      segment_command_64* segCmd = segment_command_64::make(sections.size());
      strncpy(segCmd->segname, lastSegName.data(), 16);
      segCmd->initprot = this->permissionsFromSections(sections);
      segCmd->maxprot = VM_PROT_READ|VM_PROT_WRITE|VM_PROT_EXECUTE;
      this->addLoadCommand(segCmd);
      unsigned int index = 0;
      for (SectionChunk *chunk : sections) {
        this->setMachOSection(chunk, segCmd, index);
        ++index;
      }
      // Reset to begin new segment.
      sections.clear();
      lastSegName = entryName;
    }
    sections.push_back(entry.chunk);
  }
  // Add last segment load command.
  segment_command_64* segCmd = segment_command_64::make(sections.size());
  strncpy(segCmd->segname, lastSegName.data(), 16);
  segCmd->initprot = this->permissionsFromSections(sections);;
  segCmd->maxprot = VM_PROT_READ|VM_PROT_WRITE|VM_PROT_EXECUTE;
  this->addLoadCommand(segCmd);
  unsigned int index = 0;
  for (SectionChunk *chunk : sections) {
    this->setMachOSection(chunk, segCmd, index);
    ++index;
  }

  // Add LINKEDIT segment load command
  _linkEditSegment = segment_command_64::make(0);
  strcpy(_linkEditSegment->segname, "__LINKEDIT");
  _linkEditSegment->initprot = VM_PROT_READ;
  _linkEditSegment->maxprot = VM_PROT_READ;
  this->addLoadCommand(_linkEditSegment);

  // Add dyld load command.
  this->addLoadCommand(dylinker_command::make("/usr/lib/dyld"));

  // Add dylib load commands.
  llvm::StringMap<uint32_t> dylibNamesToOrdinal;
  for (const SharedLibraryAtom* shlibAtom : file.sharedLibrary() ) {
    StringRef installName = shlibAtom->loadName();
    if ( dylibNamesToOrdinal.count(installName) == 0 ) {
      uint32_t ord = dylibNamesToOrdinal.size();
      dylibNamesToOrdinal[installName] = ord;
    }
  }
  for (llvm::StringMap<uint32_t>::iterator it=dylibNamesToOrdinal.begin(),
                            end=dylibNamesToOrdinal.end(); it != end; ++it) {
    this->addLoadCommand(dylib_command::make(it->first().data()));
  }
  
  // Add symbol table load command
  _symbolTableLoadCommand = symtab_command::make();
  this->addLoadCommand(_symbolTableLoadCommand);
 
  // Add dyld info load command
  _dyldInfoLoadCommand = dyld_info_command::make();
  this->addLoadCommand(_dyldInfoLoadCommand);
  
  // Add entry point load command
  _entryPointLoadCommand = entry_point_command::make();
  this->addLoadCommand(_entryPointLoadCommand);
    
  // Compute total size.
  _size = _mh.loadCommandsSize();
}


void LoadCommandsChunk::updateLoadCommandContent(const lld::File &file) {
  // Update segment/section information in segment load commands
  segment_command_64 *lastSegment = nullptr;
  for (ChunkSegInfo &entry : _sectionInfo) {
    // Set section info.
    ::strncpy(entry.section->sectname, entry.chunk->sectionName().data(), 16);
    ::strncpy(entry.section->segname, entry.chunk->segmentName().data(), 16);
    entry.section->addr   = entry.chunk->address();
    entry.section->size   = entry.chunk->size();
    entry.section->offset = entry.chunk->fileOffset();
    entry.section->align  = entry.chunk->align2();
    entry.section->reloff = 0;
    entry.section->nreloc = 0;
    entry.section->flags  = entry.chunk->flags();
    // Adjust segment info if needed.
    if ( entry.segment != lastSegment ) {
      // This is first section in segment.
      if ( strcmp(entry.segment->segname, "__TEXT") == 0 ) {
        // __TEXT segment is special need mach_header section.
        entry.segment->vmaddr = _writer._chunks.front()->address();
        entry.segment->fileoff = _writer._chunks.front()->fileOffset();
      }
      else {
        entry.segment->vmaddr  = entry.chunk->address();
        entry.segment->fileoff = entry.chunk->fileOffset();
      }
      
      lastSegment = entry.segment;
    }
    uint64_t sectionEndAddr = entry.section->addr + entry.section->size;
    if ( entry.segment->vmaddr + entry.segment->vmsize < sectionEndAddr) {
      uint64_t sizeToEndOfSection = sectionEndAddr - entry.segment->vmaddr;
      entry.segment->vmsize = alignTo(sizeToEndOfSection, 12);
      // zero-fill sections do not increase the segment's filesize
      if ( ! entry.chunk->occupiesNoDiskSpace() ) {
        entry.segment->filesize = alignTo(sizeToEndOfSection, 12);
      }
    }
  }
  uint64_t linkEditSize = _writer._stringsChunk->fileOffset() 
                        + _writer._stringsChunk->size()
                        - _writer._linkEditStartOffset;
  _linkEditSegment->vmaddr   = _writer._linkEditStartAddress;
  _linkEditSegment->vmsize   = alignTo(linkEditSize,12);
  _linkEditSegment->fileoff  = _writer._linkEditStartOffset;
  _linkEditSegment->filesize = linkEditSize;
  
  // Update dyld_info load command.
  _dyldInfoLoadCommand->bind_off       = _writer._bindingInfo->fileOffset();
  _dyldInfoLoadCommand->bind_size      = _writer._bindingInfo->size();
  _dyldInfoLoadCommand->lazy_bind_off  = _writer._lazyBindingInfo->fileOffset();
  _dyldInfoLoadCommand->lazy_bind_size = _writer._lazyBindingInfo->size();
  
  
  // Update symbol table load command.
  _symbolTableLoadCommand->symoff  = _writer._symbolTableChunk->fileOffset();
  _symbolTableLoadCommand->nsyms   = _writer._symbolTableChunk->count();
  _symbolTableLoadCommand->stroff  = _writer._stringsChunk->fileOffset();
  _symbolTableLoadCommand->strsize = _writer._stringsChunk->size();

  // Update entry point
  if ( _entryPointLoadCommand != nullptr ) {
    const Atom *mainAtom = file.entryPoint();
    assert(mainAtom != nullptr);
    uint32_t entryOffset = _writer.addressOfAtom(mainAtom) - _mh.address();
    _entryPointLoadCommand->entryoff = entryOffset;
  }
}


void LoadCommandsChunk::addSection(SectionChunk* chunk) {
  _sectionInfo.push_back({chunk, nullptr, nullptr});
}

void LoadCommandsChunk::addLoadCommand(load_command* lc) {
  _mh.recordLoadCommand(lc);
  _loadCmds.push_back(lc);
}



//===----------------------------------------------------------------------===//
//  LoadCommandPaddingChunk
//===----------------------------------------------------------------------===//

LoadCommandPaddingChunk::LoadCommandPaddingChunk(LoadCommandsChunk& lcc)
  : _loadCommandsChunk(lcc) {
}

StringRef LoadCommandPaddingChunk::segmentName() const {
  return StringRef("__TEXT");
}

void LoadCommandPaddingChunk::write(raw_ostream &out) {
  assert( out.tell() == _fileOffset);
  // Zero fill padding.
  this->writeZeros(_size, out);
}

const char* LoadCommandPaddingChunk::info() {
  return "padding";
}

// Segments are page sized.  Normally, any extra space not used by atoms
// is put at the end of the last page.  But the __TEXT segment is special.
// Any extra space is put between the load commands and the first section.
// The padding is put there to allow the load commands to be
// post-processed which might potentially grow them.
void LoadCommandPaddingChunk::computeSize() {
 // Layout __TEXT sections backwards from end of page to get padding up front.
  uint64_t addr = 0;
  std::vector<LoadCommandsChunk::ChunkSegInfo>& sects 
                                        = _loadCommandsChunk._sectionInfo;
  for (auto it=sects.rbegin(), end=sects.rend(); it != end; ++it) {
    LoadCommandsChunk::ChunkSegInfo &entry = *it;
    if ( !entry.chunk->segmentName().equals("__TEXT") )
      continue;
    addr -= entry.chunk->size();
    addr = addr & (0 - (1 << entry.chunk->align2()));
  }
  // Subtract out size of mach_header and all load commands.
  addr -= _loadCommandsChunk._mh.size();
  addr -= _loadCommandsChunk.size();
  // Modulo page size to get padding needed between load commands 
  // and first section.
  _size = (addr % 4096); 
}

//===----------------------------------------------------------------------===//
//  LinkEditChunk
//===----------------------------------------------------------------------===//

LinkEditChunk::LinkEditChunk() {
  _align2 = 3;
}

StringRef LinkEditChunk::segmentName() const {
  return StringRef("__LINKEDIT");
}

  
//===----------------------------------------------------------------------===//
//  DyldInfoChunk
//===----------------------------------------------------------------------===//
DyldInfoChunk::DyldInfoChunk(MachOWriter &writer)
 : _writer(writer) {
}

void DyldInfoChunk::write(raw_ostream &out) {
  assert( out.tell() == _fileOffset);
  for ( uint8_t byte : _bytes ) {
    out.write(byte);
  }
}

void DyldInfoChunk::append_byte(uint8_t b) {
  _bytes.push_back(b);
}

void DyldInfoChunk::append_string(StringRef str) {
  const char *s = str.data();
  for (int i=str.size(); i > 0; --i) {
    _bytes.push_back(*s++);
  }
  _bytes.push_back('\0');
}

void DyldInfoChunk::append_uleb128(uint64_t value) {
  uint8_t byte;
  do {
    byte = value & 0x7F;
    value &= ~0x7F;
    if ( value != 0 )
      byte |= 0x80;
    _bytes.push_back(byte);
    value = value >> 7;
  } while( byte >= 0x80 );
}



//===----------------------------------------------------------------------===//
//  BindingInfoChunk
//===----------------------------------------------------------------------===//

BindingInfoChunk::BindingInfoChunk(MachOWriter &writer)
 : DyldInfoChunk(writer) {
}

const char* BindingInfoChunk::info() {
  return "binding info";
}

void BindingInfoChunk::computeSize(const lld::File &file,
                                    const std::vector<SectionChunk*> &chunks) {
  for (const SectionChunk *chunk : chunks ) {
    // skip lazy pointer section
    if ( chunk->flags() == S_LAZY_SYMBOL_POINTERS )
      continue;
    // skip code sections
    if ( chunk->flags() == (S_REGULAR | S_ATTR_PURE_INSTRUCTIONS) )
      continue;
    uint64_t segStartAddr = 0;
    uint64_t segEndAddr = 0;
    uint32_t segIndex = 0;
    _writer.findSegment(chunk->segmentName(),
                                    &segIndex, &segStartAddr, &segEndAddr);
    for (const SectionChunk::AtomInfo &info : chunk->atoms() ) {
      const DefinedAtom* atom = info.atom;
      StringRef targetName;
      int ordinal;
  
      // look for fixups pointing to shlib atoms
      for (const Reference *ref : *atom ) {
        const Atom *target = ref->target();
        if ( target != nullptr ) {
          const SharedLibraryAtom *shlTarget 
                                        = dyn_cast<SharedLibraryAtom>(target);
          if ( shlTarget != nullptr ) {
            assert(ref->kind() == ReferenceKind::pointer64);
            targetName = shlTarget->name();
            ordinal = 1;
          }
        }
      }
      
      if ( targetName.empty() )
        continue;
      
      // write location of fixup
      this->append_byte(BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB | segIndex);
      uint64_t address = _writer.addressOfAtom(atom);
      this->append_uleb128(address - segStartAddr);

      // write ordinal
      if ( ordinal <= 0 ) {
        // special lookups are encoded as negative numbers in BindingInfo
        this->append_byte(BIND_OPCODE_SET_DYLIB_SPECIAL_IMM
                                          | (ordinal & BIND_IMMEDIATE_MASK) );
      }
      else if ( ordinal <= 15 ) {
        // small ordinals are encoded in opcode
        this->append_byte(BIND_OPCODE_SET_DYLIB_ORDINAL_IMM | ordinal);
      }
      else {
        this->append_byte(BIND_OPCODE_SET_DYLIB_ORDINAL_ULEB);
        this->append_uleb128(ordinal);
      }
      
      // write binding type
      this->append_byte(BIND_OPCODE_SET_TYPE_IMM | BIND_TYPE_POINTER);
      
      // write symbol name and flags
      int flags = 0;
      this->append_byte(BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM | flags);
      this->append_string(targetName);

      // write do bind
      this->append_byte(BIND_OPCODE_DO_BIND);
      this->append_byte(BIND_OPCODE_DONE);
    }
  }
  _size = _bytes.size();
}


//===----------------------------------------------------------------------===//
//  LazyBindingInfoChunk
//===----------------------------------------------------------------------===//

LazyBindingInfoChunk::LazyBindingInfoChunk(MachOWriter &writer)
 : DyldInfoChunk(writer) {
}

const char* LazyBindingInfoChunk::info() {
  return "lazy binding info";
}

void LazyBindingInfoChunk::updateHelper(const DefinedAtom *lazyPointerAtom,
                                        uint32_t offset) {
  for (const Reference *ref : *lazyPointerAtom ) {
    if ( ref->kind() != ReferenceKind::pointer64 )
      continue;
    const DefinedAtom *helperAtom = dyn_cast<DefinedAtom>(ref->target());
    assert(helperAtom != nullptr);
    for (const Reference *href : *helperAtom ) {
      if ( href->kind() == ReferenceKind::lazyImm ) {
        (const_cast<Reference*>(href))->setAddend(offset);
        return;
      }
    }
  }
  assert(0 && "could not update helper lazy immediate value");
}

void LazyBindingInfoChunk::computeSize(const lld::File &file,
                                    const std::vector<SectionChunk*> &chunks) {
  for (const SectionChunk *chunk : chunks ) {
    if ( chunk->flags() != S_LAZY_SYMBOL_POINTERS )
      continue;
    uint64_t segStartAddr = 0;
    uint64_t segEndAddr = 0;
    uint32_t segIndex = 0;
    _writer.findSegment(chunk->segmentName(),
                                    &segIndex, &segStartAddr, &segEndAddr);
    for (const SectionChunk::AtomInfo &info : chunk->atoms() ) {
      const DefinedAtom *lazyPointerAtom = info.atom;
      assert(lazyPointerAtom->contentType() == DefinedAtom::typeLazyPointer);
      // Update help to have offset of the lazy binding info.
      this->updateHelper(lazyPointerAtom, _bytes.size());

      // Write location of fixup.
      this->append_byte(BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB | segIndex);
      uint64_t address = _writer.addressOfAtom(lazyPointerAtom);
      this->append_uleb128(address - segStartAddr);

      // write ordinal
      int ordinal = 1;
      if ( ordinal <= 0 ) {
        // special lookups are encoded as negative numbers in BindingInfo
        this->append_byte(BIND_OPCODE_SET_DYLIB_SPECIAL_IMM
                                          | (ordinal & BIND_IMMEDIATE_MASK) );
      }
      else if ( ordinal <= 15 ) {
        // small ordinals are encoded in opcode
        this->append_byte(BIND_OPCODE_SET_DYLIB_ORDINAL_IMM | ordinal);
      }
      else {
        this->append_byte(BIND_OPCODE_SET_DYLIB_ORDINAL_ULEB);
        this->append_uleb128(ordinal);
      }
      
      // write symbol name and flags
      int flags = 0;
      StringRef name;
      for (const Reference *ref : *lazyPointerAtom ) {
        if ( ref->kind() == ReferenceKind::lazyTarget ) {
          const Atom *shlib = ref->target();
          assert(shlib != nullptr);
          name = shlib->name();
        }
      }
      assert(!name.empty());
      this->append_byte(BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM | flags);
      this->append_string(name);

      // write do bind
      this->append_byte(BIND_OPCODE_DO_BIND);
      this->append_byte(BIND_OPCODE_DONE);
    }
  }
  _size = _bytes.size();
}


//===----------------------------------------------------------------------===//
//  SymbolTableChunk
//===----------------------------------------------------------------------===//

SymbolTableChunk::SymbolTableChunk(SymbolStringsChunk& str) 
  : _stringsChunk(str) {
}

void SymbolTableChunk::write(raw_ostream &out) {
  assert( out.tell() == _fileOffset);
  for ( nlist_64 &sym : _globalDefinedsymbols ) {
    sym.write(out);
  }
  for ( nlist_64 &sym : _localDefinedsymbols ) {
    sym.write(out);
  }
  for ( nlist_64 &sym : _undefinedsymbols ) {
    sym.write(out);
  }
}

const char* SymbolTableChunk::info() {
  return "symbol tables ";
}

uint32_t SymbolTableChunk::count() {
  return _globalDefinedsymbols.size() 
       + _localDefinedsymbols.size()
       + _undefinedsymbols.size();
}

uint8_t SymbolTableChunk::nType(const DefinedAtom *atom) {
  uint8_t result = N_SECT;
  switch ( atom->scope() ) {
    case DefinedAtom::scopeTranslationUnit:
      break;
    case DefinedAtom::scopeLinkageUnit:
      result |= N_EXT | N_PEXT;
      break;
    case DefinedAtom::scopeGlobal:
      result |= N_EXT;
      break;
  }
  return result;
}

void SymbolTableChunk::computeSize(const lld::File &file,
                                   const std::vector<SectionChunk*> &chunks) {
  // Add symbols for definitions
  unsigned int sectionIndex = 1;
  for (const SectionChunk *chunk : chunks ) {
    for (const SectionChunk::AtomInfo &info : chunk->atoms() ) {
      if ( info.atom->name().empty() )
        continue;
      uint64_t atomAddress = chunk->address() + info.offsetInSection;
      nlist_64 sym;
      sym.n_strx = _stringsChunk.stringIndex(info.atom->name());
      sym.n_type = this->nType(info.atom);
      sym.n_sect = sectionIndex;
      sym.n_value = atomAddress;
      if ( info.atom->scope() == DefinedAtom::scopeGlobal )
        _globalDefinedsymbols.push_back(sym);
      else
        _localDefinedsymbols.push_back(sym);
    }
    ++sectionIndex;
  }
  
  // Add symbols for undefined/sharedLibrary symbols
  for (const SharedLibraryAtom* atom : file.sharedLibrary() ) {
    nlist_64 sym;
    sym.n_strx = _stringsChunk.stringIndex(atom->name());
    sym.n_type = N_UNDF;
    sym.n_sect = 0;
    sym.n_value = 0;
    _undefinedsymbols.push_back(sym);
  }
  
  _size = sizeof(nlist_64) * this->count();
}


//===----------------------------------------------------------------------===//
//  SymbolStringsChunk
//===----------------------------------------------------------------------===//

SymbolStringsChunk::SymbolStringsChunk() {
  // mach-o reserves the first byte in the string pool so that 
  // zero is never a valid string index.
  _strings.push_back('\0');
}


void SymbolStringsChunk::write(raw_ostream &out) {
  assert( out.tell() == _fileOffset);
  for ( char c : _strings ) {
    out.write(c);
  }
}

const char* SymbolStringsChunk::info() {
  return "symbol strings ";
}

void SymbolStringsChunk::computeSize(const lld::File &file,
                                     const std::vector<SectionChunk*>&) {
  _size = _strings.size();
}


uint32_t SymbolStringsChunk::stringIndex(StringRef str) {
  uint32_t result = _strings.size();
  const char* s = str.data();
  for (int i=0; i < str.size(); ++i) {
    _strings.push_back(s[i]);
  }
  _strings.push_back('\0');
  return result;
}


//===----------------------------------------------------------------------===//
//  MachOWriter
//===----------------------------------------------------------------------===//

MachOWriter::MachOWriter(DarwinPlatform &platform)
  : _platform(platform), _bindingInfo(nullptr), _lazyBindingInfo(nullptr),
    _symbolTableChunk(nullptr), _stringsChunk(nullptr),
    _linkEditStartOffset(0), _linkEditStartAddress(0) {
}
 
void MachOWriter::build(const lld::File &file) {
  // Create objects for each chunk.
  this->createChunks(file);
  
  // Now that SectionChunks have sizes, load commands can be laid out
  _loadCommandsChunk->computeSize(file);
 
  // Now that load commands are sized, padding can be computed
  _paddingChunk->computeSize();
  
  // Now that all chunks (except linkedit) have sizes, assign file offsets
  this->assignFileOffsets();
  
  // Now chunks have file offsets each atom can be assigned an address
  this->buildAtomToAddressMap();
  
  // Now that atoms have address, symbol table can be build
  this->buildLinkEdit(file);
  
  // Assign file offsets to linkedit chunks
  this->assignLinkEditFileOffsets();
  
  // Finally, update load commands to reflect linkEdit layout
  _loadCommandsChunk->updateLoadCommandContent(file);
}


void MachOWriter::createChunks(const lld::File &file) {
  // Assign atoms to chunks, creating new chunks as needed
  std::map<DefinedAtom::ContentType, SectionChunk*> map;
  for (const DefinedAtom* atom : file.defined() ) {
    assert( atom->sectionChoice() == DefinedAtom::sectionBasedOnContent );
    DefinedAtom::ContentType type = atom->contentType();
    auto pos = map.find(type);
    if ( pos == map.end() ) {
      SectionChunk *chunk = SectionChunk::make(type, _platform, *this);
      map[type] = chunk;
      chunk->appendAtom(atom);
    }
    else {
      pos->second->appendAtom(atom);
    }
  }
  
  // Sort Chunks so ones in same segment are contiguous.
  
  
  // Make chunks in __TEXT for mach_header and load commands at start.
  MachHeaderChunk *mhc = new MachHeaderChunk(_platform, file);
  _chunks.push_back(mhc);
  
  _loadCommandsChunk = new LoadCommandsChunk(*mhc, _platform, *this);
  _chunks.push_back(_loadCommandsChunk);
  
  _paddingChunk = new LoadCommandPaddingChunk(*_loadCommandsChunk);
  _chunks.push_back(_paddingChunk);
  
  for (auto it=map.begin(); it != map.end(); ++it) {
     _chunks.push_back(it->second);
     _sectionChunks.push_back(it->second);
     _loadCommandsChunk->addSection(it->second);
  }
  
  // Make LINKEDIT chunks.
  _bindingInfo = new BindingInfoChunk(*this);
  _lazyBindingInfo = new LazyBindingInfoChunk(*this);
  _stringsChunk = new SymbolStringsChunk();
  _symbolTableChunk = new SymbolTableChunk(*_stringsChunk);
  this->addLinkEditChunk(_bindingInfo);
  this->addLinkEditChunk(_lazyBindingInfo);
  this->addLinkEditChunk(_symbolTableChunk);
  this->addLinkEditChunk(_stringsChunk);
}
 

void MachOWriter::addLinkEditChunk(LinkEditChunk *chunk) {
  _linkEditChunks.push_back(chunk);
  _chunks.push_back(chunk);
}


void MachOWriter::buildAtomToAddressMap() {
  DEBUG(llvm::dbgs() << "assign atom addresses:\n");
  for (SectionChunk *chunk : _sectionChunks ) {
    for (const SectionChunk::AtomInfo &info : chunk->atoms() ) {
      _atomToAddress[info.atom] = chunk->address() + info.offsetInSection;
      DEBUG(llvm::dbgs() << "   address=0x";
            llvm::dbgs().write_hex(_atomToAddress[info.atom]);
            llvm::dbgs() << " atom=" << info.atom;
            llvm::dbgs() << " name=" << info.atom->name() << "\n");
    }
  }
}


//void MachOWriter::dump() {
//  for ( Chunk *chunk : _chunks ) {
//    fprintf(stderr, "size=0x%08llX, fileOffset=0x%08llX, address=0x%08llX %s\n",
//          chunk->size(), chunk->fileOffset(),chunk->address(), chunk->info());
//  }
//}

void MachOWriter::assignFileOffsets() {
  DEBUG(llvm::dbgs() << "assign file offsets:\n");
  uint64_t offset = 0;
  uint64_t address = _platform.pageZeroSize();
  for ( Chunk *chunk : _chunks ) {
    if ( chunk->segmentName().equals("__LINKEDIT") ) {
      _linkEditStartOffset  = Chunk::alignTo(offset, 12);
      _linkEditStartAddress = Chunk::alignTo(address, 12);
      break;
    }
    chunk->assignFileOffset(offset, address);
  }
}

void MachOWriter::assignLinkEditFileOffsets() {
  DEBUG(llvm::dbgs() << "assign LINKEDIT file offsets:\n");
  uint64_t offset = _linkEditStartOffset;
  uint64_t address = _linkEditStartAddress;
  for ( Chunk *chunk : _linkEditChunks ) {
    chunk->assignFileOffset(offset, address);
  }
}

void MachOWriter::buildLinkEdit(const lld::File &file) {
  for (LinkEditChunk *chunk : _linkEditChunks) {
    chunk->computeSize(file, _sectionChunks);
  }
}


uint64_t MachOWriter::addressOfAtom(const Atom *atom) {
  return _atomToAddress[atom];
}


void MachOWriter::findSegment(StringRef segmentName, uint32_t *segIndex,
                                uint64_t *segStartAddr, uint64_t *segEndAddr) {
  const uint64_t kInvalidAddress = (uint64_t)(-1);
  StringRef lastSegName("__TEXT");
  *segIndex = 0;
  if ( _platform.pageZeroSize() != 0 ) {
      *segIndex = 1;
  }
  *segStartAddr = kInvalidAddress;
  *segEndAddr = kInvalidAddress;
  for (SectionChunk *chunk : _sectionChunks ) {
    if ( ! lastSegName.equals(chunk->segmentName()) ) {
      *segIndex += 1;
      lastSegName = chunk->segmentName();
    }
    if ( chunk->segmentName().equals(segmentName) ) {
      uint64_t  chunkEndAddr = chunk->address() + chunk->size();
      if ( *segStartAddr == kInvalidAddress ) {
        *segStartAddr = chunk->address();
        *segEndAddr = chunkEndAddr;
      }
      else if ( *segEndAddr < chunkEndAddr ) {
        *segEndAddr = chunkEndAddr;
      }
    }

  }
}


void MachOWriter::zeroFill(int64_t amount, raw_ostream &out) {
  for( int i=amount; i > 0; --i)
    out.write('\0');
}


void MachOWriter::write(raw_ostream &out) {
  for ( Chunk *chunk : _chunks ) {
    if ( out.tell() != chunk->fileOffset() ) {
      // Assume just alignment padding to start of next section.
      assert( out.tell() < chunk->fileOffset() );
      uint64_t padding = chunk->fileOffset() - out.tell();
      chunk->writeZeros(padding, out);
    }
    chunk->write(out);
  }
}


//
// Creates a mach-o final linked image from the given atom graph and writes
// it to the supplied output stream.
//
void writeExecutable(const lld::File &file, DarwinPlatform &platform, 
                                            raw_ostream &out) {
  MachOWriter writer(platform);
  writer.build(file);
  writer.write(out);
}



} // namespace darwin
} // namespace lld

