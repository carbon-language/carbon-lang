//===- lib/ReaderWriter/PECOFF/WriterPECOFF.cpp ---------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// PE/COFF file consists of DOS Header, PE Header, COFF Header and Section
/// Tables followed by raw section data.
///
/// This writer is responsible for writing Core Linker results to an Windows
/// executable file. Currently it can only output ".text" section; other
/// sections including the symbol table are silently ignored.
///
/// This writer currently supports 32 bit PE/COFF for x86 processor only.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "WriterPECOFF"

#include <algorithm>
#include <map>
#include <time.h>
#include <vector>

#include "Atoms.h"

#include "lld/Core/DefinedAtom.h"
#include "lld/Core/File.h"
#include "lld/ReaderWriter/AtomLayout.h"
#include "lld/ReaderWriter/PECOFFLinkingContext.h"
#include "lld/ReaderWriter/Writer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/Format.h"

using llvm::support::ulittle16_t;
using llvm::support::ulittle32_t;

namespace lld {
namespace pecoff {

namespace {
class SectionChunk;

// Page size of x86 processor. Some data needs to be aligned at page boundary
// when loaded into memory.
const int PAGE_SIZE = 4096;

// Disk sector size. Some data needs to be aligned at disk sector boundary in
// file.
const int SECTOR_SIZE = 512;

/// A Chunk is an abstrace contiguous range in an output file.
class Chunk {
public:
  enum Kind {
    kindHeader,
    kindSection,
    kindDataDirectory
  };

  explicit Chunk(Kind kind) : _kind(kind), _size(0), _align(1) {}
  virtual ~Chunk() {};
  virtual void write(uint8_t *fileBuffer) = 0;

  virtual uint64_t fileOffset() const { return _fileOffset; }
  virtual uint64_t size() const { return _size; }
  virtual uint64_t align() const { return _align; }

  virtual void setFileOffset(uint64_t fileOffset) {
    _fileOffset = fileOffset;
  }

  Kind getKind() const { return _kind; }

protected:
  Kind _kind;
  uint64_t _size;
  uint64_t _fileOffset;
  uint64_t _align;
};

/// A HeaderChunk is an abstract class to represent a file header for
/// PE/COFF. The data in the header chunk is metadata about program and will
/// be consumed by the windows loader. HeaderChunks are not mapped to memory
/// when executed.
class HeaderChunk : public Chunk {
public:
  HeaderChunk() : Chunk(kindHeader) {}

  static bool classof(const Chunk *c) { return c->getKind() == kindHeader; }
};

/// A DOSStubChunk represents the DOS compatible header at the beginning
/// of PE/COFF files.
class DOSStubChunk : public HeaderChunk {
public:
  DOSStubChunk(const PECOFFLinkingContext &ctx)
      : HeaderChunk(), _context(ctx) {
    // Minimum size of DOS stub is 64 bytes. The next block (PE header) needs to
    // be aligned on 8 byte boundary.
    _size = std::max(_context.getDosStub().size(), (size_t)64);
    _size = llvm::RoundUpToAlignment(_size, 8);
  }

  virtual void write(uint8_t *fileBuffer) {
    ArrayRef<uint8_t> array = _context.getDosStub();
    std::memcpy(fileBuffer, array.data(), array.size());
    auto *header = reinterpret_cast<llvm::object::dos_header *>(fileBuffer);
    header->AddressOfNewExeHeader = _size;
  }

private:
  const PECOFFLinkingContext &_context;
};

/// A PEHeaderChunk represents PE header including COFF header.
class PEHeaderChunk : public HeaderChunk {
public:
  explicit PEHeaderChunk(const PECOFFLinkingContext &context);

  virtual void write(uint8_t *fileBuffer);

  virtual void setSizeOfHeaders(uint64_t size) {
    // Must be multiple of FileAlignment.
    _peHeader.SizeOfHeaders = llvm::RoundUpToAlignment(size, SECTOR_SIZE);
  }

  virtual void setSizeOfCode(uint64_t size) {
    _peHeader.SizeOfCode = size;
  }

  virtual void setSizeOfInitializedData(uint64_t size) {
    _peHeader.SizeOfInitializedData = size;
  }

  virtual void setSizeOfUninitializedData(uint64_t size) {
    _peHeader.SizeOfUninitializedData = size;
  }

  virtual void setNumberOfSections(uint32_t num) {
    _coffHeader.NumberOfSections = num;
  }

  virtual void setBaseOfCode(uint32_t rva) { _peHeader.BaseOfCode = rva; }

  virtual void setBaseOfData(uint32_t rva) { _peHeader.BaseOfData = rva; }

  virtual void setSizeOfImage(uint32_t size) { _peHeader.SizeOfImage = size; }

  virtual void setAddressOfEntryPoint(uint32_t address) {
    _peHeader.AddressOfEntryPoint = address;
  }

private:
  llvm::object::coff_file_header _coffHeader;
  llvm::object::pe32_header _peHeader;
};

/// A SectionHeaderTableChunk represents Section Table Header of PE/COFF
/// format, which is a list of section headers.
class SectionHeaderTableChunk : public HeaderChunk {
public:
  SectionHeaderTableChunk() : HeaderChunk() {}
  void addSection(SectionChunk *chunk);
  virtual uint64_t size() const;
  virtual void write(uint8_t *fileBuffer);

private:
  std::vector<SectionChunk *> _sections;
};

/// An AtomChunk represents a section containing atoms.
class AtomChunk : public Chunk {
public:
  virtual void write(uint8_t *fileBuffer);

  /// Add all atoms to the given map. This data will be used to do relocation.
  void buildAtomToVirtualAddr(std::map<const Atom *, uint64_t> &atomRva) {
    for (const auto *layout : _atomLayouts)
      atomRva[layout->_atom] = layout->_virtualAddr;
  }

  void applyRelocations(uint8_t *fileBuffer,
                        std::map<const Atom *, uint64_t> &atomRva,
                        uint64_t imageBaseAddress);
  void printAtomAddresses(uint64_t baseAddr);
  void addBaseRelocations(std::vector<uint64_t> &relocSites);

  // Set the file offset of the beginning of this section.
  virtual void setFileOffset(uint64_t fileOffset) {
    Chunk::setFileOffset(fileOffset);
    for (AtomLayout *layout : _atomLayouts)
      layout->_fileOffset += fileOffset;
  }

  uint64_t getSectionRva() {
    assert(_atomLayouts.size() > 0);
    return _atomLayouts[0]->_virtualAddr;
  }

  virtual void setVirtualAddress(uint32_t rva) {
    for (AtomLayout *layout : _atomLayouts)
      layout->_virtualAddr += rva;
  }

  uint64_t getAtomVirtualAddress(StringRef name) {
    for (auto atomLayout : _atomLayouts)
      if (atomLayout->_atom->name() == name)
        return atomLayout->_virtualAddr;
    return 0;
  }

  static bool classof(const Chunk *c) {
    Kind kind = c->getKind();
    return kind == kindSection || kind == kindDataDirectory;
  }

protected:
  AtomChunk(Kind kind) : Chunk(kind) {}
  std::vector<AtomLayout *> _atomLayouts;
};

/// A DataDirectoryChunk represents data directory entries that follows the PE
/// header in the output file. An entry consists of an 8 byte field that
/// indicates a relative virtual address (the starting address of the entry data
/// in memory) and 8 byte entry data size.
class DataDirectoryChunk : public AtomChunk {
public:
  DataDirectoryChunk(const DefinedAtom *atom) : AtomChunk(kindDataDirectory) {
    if (atom)
      _atomLayouts.push_back(new (_alloc) AtomLayout(atom, 0, 0));
  }

  virtual uint64_t size() const {
    return sizeof(llvm::object::data_directory) * 16;
  }

  void setBaseRelocField(uint32_t addr, uint32_t size) {
    _baseRelocAddr = addr;
    _baseRelocSize = size;
  }

  virtual void write(uint8_t *fileBuffer);

private:
  uint32_t _baseRelocAddr;
  uint32_t _baseRelocSize;
  mutable llvm::BumpPtrAllocator _alloc;
};

/// A SectionChunk represents a section containing atoms. It consists of a
/// section header that to be written to PECOFF header and atoms which to be
/// written to the raw data section.
class SectionChunk : public AtomChunk {
public:
  /// Returns the size of the section on disk. The returned value is multiple
  /// of disk sector, so the size may include the null padding at the end of
  /// section.
  virtual uint64_t size() const {
    return llvm::RoundUpToAlignment(_size, _align);
  }

  virtual uint64_t rawSize() const {
    return _size;
  }

  // Set the file offset of the beginning of this section.
  virtual void setFileOffset(uint64_t fileOffset) {
    AtomChunk::setFileOffset(fileOffset);
    _sectionHeader.PointerToRawData = fileOffset;
  }

  virtual void setVirtualAddress(uint32_t rva) {
    _sectionHeader.VirtualAddress = rva;
    AtomChunk::setVirtualAddress(rva);
  }

  virtual uint32_t getVirtualAddress() { return _sectionHeader.VirtualAddress; }

  virtual llvm::object::coff_section &getSectionHeader();

  ulittle32_t getSectionCharacteristics();

  void appendAtom(const DefinedAtom *atom);

  static bool classof(const Chunk *c) { return c->getKind() == kindSection; }

protected:
  SectionChunk(StringRef sectionName, uint32_t characteristics);

  void buildContents(const File &linkedFile,
                     bool (*isEligible)(const DefinedAtom *));
  const uint32_t _characteristics;

  llvm::object::coff_section _sectionHeader;

private:
  llvm::object::coff_section
  createSectionHeader(StringRef sectionName, uint32_t characteristics) const;

  mutable llvm::BumpPtrAllocator _alloc;
};

// \brief A GenericSectionChunk represents various sections such as .text or
// .data.
class GenericSectionChunk : public SectionChunk {
public:
  virtual void write(uint8_t *fileBuffer);

  GenericSectionChunk(const PECOFFLinkingContext &ctx, StringRef name,
                      const std::vector<const DefinedAtom *> &atoms)
      : SectionChunk(name, getCharacteristics(ctx, name, atoms)) {
    for (auto *a : atoms)
      appendAtom(a);
    _sectionHeader.VirtualSize = _size;
    _sectionHeader.SizeOfRawData = size();
  }

private:
  uint32_t getCharacteristics(const PECOFFLinkingContext &ctx, StringRef name,
                              const std::vector<const DefinedAtom *> &atoms) {
    return ctx.getSectionAttributes(name, getDefaultCharacteristics(name, atoms));
  }

  uint32_t getDefaultCharacteristics(
      StringRef name, const std::vector<const DefinedAtom *> &atoms);
};

/// A BaseRelocAtom represents a base relocation block in ".reloc" section.
class BaseRelocAtom : public coff::COFFLinkerInternalAtom {
public:
  BaseRelocAtom(const File &file, uint64_t ordinal, std::vector<uint8_t> data)
      : COFFLinkerInternalAtom(file, ordinal, std::move(data)) {}

  virtual ContentType contentType() const { return typeData; }
  virtual Alignment alignment() const { return Alignment(2); }
};

/// A BaseRelocChunk represents ".reloc" section.
///
/// .reloc section contains a list of addresses. If the PE/COFF loader decides
/// to load the binary at a memory address different from its preferred base
/// address, which is specified by ImageBase field in the COFF header, the
/// loader needs to relocate the binary, so that all the addresses in the binary
/// point to new locations. The loader will do that by fixing up the addresses
/// specified by .reloc section.
///
/// The executable is almost always loaded at the preferred base address because
/// it's loaded into an empty address space. The DLL is however an subject of
/// load-time relocation because it may conflict with other DLLs or the
/// executable.
class BaseRelocChunk : public SectionChunk {
  typedef std::vector<std::unique_ptr<Chunk>> ChunkVectorT;
  typedef std::map<uint64_t, std::vector<uint16_t>> PageOffsetT;

public:
  BaseRelocChunk(const File &linkedFile)
      : SectionChunk(".reloc", characteristics), _file(linkedFile) {}

  void setContents(ChunkVectorT &chunks);

private:
  // When loaded into memory, reloc section should be readable and writable.
  static const uint32_t characteristics =
      llvm::COFF::IMAGE_SCN_MEM_READ |
      llvm::COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
      llvm::COFF::IMAGE_SCN_MEM_DISCARDABLE;

  // Returns a list of RVAs that needs to be relocated if the binary is loaded
  // at an address different from its preferred one.
  std::vector<uint64_t> listRelocSites(ChunkVectorT &chunks);

  // Divide the given RVAs into blocks.
  PageOffsetT groupByPage(std::vector<uint64_t> relocSites);

  // Create the content of a relocation block.
  DefinedAtom *createBaseRelocBlock(const File &file, uint64_t ordinal,
                                    uint64_t pageAddr,
                                    const std::vector<uint16_t> &offsets);

  mutable llvm::BumpPtrAllocator _alloc;
  const File &_file;
};

PEHeaderChunk::PEHeaderChunk(const PECOFFLinkingContext &context)
    : HeaderChunk() {
  // Set the size of the chunk and initialize the header with null bytes.
  _size = sizeof(llvm::COFF::PEMagic) + sizeof(_coffHeader) + sizeof(_peHeader);
  std::memset(&_coffHeader, 0, sizeof(_coffHeader));
  std::memset(&_peHeader, 0, sizeof(_peHeader));

  _coffHeader.Machine = context.getMachineType();
  _coffHeader.TimeDateStamp = time(NULL);

  // The size of PE header including optional data directory is always 224.
  _coffHeader.SizeOfOptionalHeader = 224;

  // Attributes of the executable.
  uint16_t characteristics = llvm::COFF::IMAGE_FILE_32BIT_MACHINE |
                             llvm::COFF::IMAGE_FILE_EXECUTABLE_IMAGE;
  if (context.getLargeAddressAware())
    characteristics |= llvm::COFF::IMAGE_FILE_LARGE_ADDRESS_AWARE;
  if (context.getSwapRunFromCD())
    characteristics |= llvm::COFF::IMAGE_FILE_REMOVABLE_RUN_FROM_SWAP;
  if (context.getSwapRunFromNet())
    characteristics |= llvm::COFF::IMAGE_FILE_NET_RUN_FROM_SWAP;
  if (!context.getBaseRelocationEnabled())
    characteristics |= llvm::COFF::IMAGE_FILE_RELOCS_STRIPPED;

  _coffHeader.Characteristics = characteristics;

  // 0x10b indicates a normal PE32 executable. For PE32+ it should be 0x20b.
  _peHeader.Magic = 0x10b;

  // The address of entry point relative to ImageBase. Windows executable
  // usually starts at address 0x401000.
  _peHeader.AddressOfEntryPoint = 0x1000;

  // The address of the executable when loaded into memory. The default for
  // DLLs is 0x10000000. The default for executables is 0x400000.
  _peHeader.ImageBase = context.getBaseAddress();

  // Sections should be page-aligned when loaded into memory, which is 4KB on
  // x86.
  _peHeader.SectionAlignment = context.getSectionDefaultAlignment();

  // Sections in an executable file on disk should be sector-aligned (512 byte).
  _peHeader.FileAlignment = SECTOR_SIZE;

  // The version number of the resultant executable/DLL. The number is purely
  // informative, and neither the linker nor the loader won't use it. User can
  // set the value using /version command line option. Default is 0.0.
  PECOFFLinkingContext::Version imageVersion = context.getImageVersion();
  _peHeader.MajorImageVersion = imageVersion.majorVersion;
  _peHeader.MinorImageVersion = imageVersion.minorVersion;

  // The required Windows version number. This is the internal version and
  // shouldn't be confused with product name. Windows 7 is version 6.1 and
  // Windows 8 is 6.2, for example.
  PECOFFLinkingContext::Version minOSVersion = context.getMinOSVersion();
  _peHeader.MajorOperatingSystemVersion = minOSVersion.majorVersion;
  _peHeader.MinorOperatingSystemVersion = minOSVersion.minorVersion;
  _peHeader.MajorSubsystemVersion = minOSVersion.majorVersion;
  _peHeader.MinorSubsystemVersion = minOSVersion.minorVersion;

  _peHeader.Subsystem = context.getSubsystem();

  // Despite its name, DLL characteristics field has meaning both for
  // executables and DLLs. We are not very sure if the following bits must
  // be set, but regular binaries seem to have these bits, so we follow
  // them.
  uint16_t dllCharacteristics = llvm::COFF::IMAGE_DLL_CHARACTERISTICS_NO_SEH;
  if (context.isTerminalServerAware())
    dllCharacteristics |=
        llvm::COFF::IMAGE_DLL_CHARACTERISTICS_TERMINAL_SERVER_AWARE;
  if (context.isNxCompat())
    dllCharacteristics |= llvm::COFF::IMAGE_DLL_CHARACTERISTICS_NX_COMPAT;
  if (context.getDynamicBaseEnabled())
    dllCharacteristics |= llvm::COFF::IMAGE_DLL_CHARACTERISTICS_DYNAMIC_BASE;
  if (!context.getAllowBind())
    dllCharacteristics |= llvm::COFF::IMAGE_DLL_CHARACTERISTICS_NO_BIND;
  if (!context.getAllowIsolation())
    dllCharacteristics |= llvm::COFF::IMAGE_DLL_CHARACTERISTICS_NO_ISOLATION;
  _peHeader.DLLCharacteristics = dllCharacteristics;

  _peHeader.SizeOfStackReserve = context.getStackReserve();
  _peHeader.SizeOfStackCommit = context.getStackCommit();
  _peHeader.SizeOfHeapReserve = context.getHeapReserve();
  _peHeader.SizeOfHeapCommit = context.getHeapCommit();

  // The number of data directory entries. We always have 16 entries.
  _peHeader.NumberOfRvaAndSize = 16;
}

void PEHeaderChunk::write(uint8_t *fileBuffer) {
  fileBuffer += fileOffset();
  std::memcpy(fileBuffer, llvm::COFF::PEMagic, sizeof(llvm::COFF::PEMagic));
  fileBuffer += sizeof(llvm::COFF::PEMagic);
  std::memcpy(fileBuffer, &_coffHeader, sizeof(_coffHeader));
  fileBuffer += sizeof(_coffHeader);
  std::memcpy(fileBuffer, &_peHeader, sizeof(_peHeader));
}

void AtomChunk::write(uint8_t *fileBuffer) {
  for (const auto *layout : _atomLayouts) {
    const DefinedAtom *atom = cast<DefinedAtom>(layout->_atom);
    ArrayRef<uint8_t> rawContent = atom->rawContent();
    std::memcpy(fileBuffer + layout->_fileOffset, rawContent.data(),
                rawContent.size());
  }
}

void AtomChunk::applyRelocations(uint8_t *fileBuffer,
                                 std::map<const Atom *, uint64_t> &atomRva,
                                 uint64_t imageBaseAddress) {
  for (const auto *layout : _atomLayouts) {
    const DefinedAtom *atom = cast<DefinedAtom>(layout->_atom);
    for (const Reference *ref : *atom) {
      auto relocSite = reinterpret_cast<ulittle32_t *>(
          fileBuffer + layout->_fileOffset + ref->offsetInAtom());
      uint64_t targetAddr = atomRva[ref->target()];
      // Also account for whatever offset is already stored at the relocation
      // site.
      targetAddr += *relocSite;

      // Skip if this reference is not for relocation.
      if (ref->kind() < lld::Reference::kindTargetLow)
        continue;

      switch (ref->kind()) {
      case llvm::COFF::IMAGE_REL_I386_ABSOLUTE:
        // This relocation is no-op.
        break;
      case llvm::COFF::IMAGE_REL_I386_DIR32:
        // Set target's 32-bit VA.
        *relocSite = targetAddr + imageBaseAddress;
        break;
      case llvm::COFF::IMAGE_REL_I386_DIR32NB:
        // Set target's 32-bit RVA.
        *relocSite = targetAddr;
        break;
      case llvm::COFF::IMAGE_REL_I386_REL32: {
        // Set 32-bit relative address of the target. This relocation is
        // usually used for relative branch or call instruction.
        uint32_t disp = atomRva[atom] + ref->offsetInAtom() + 4;
        *relocSite = targetAddr - disp;
        break;
      }
      default:
        llvm_unreachable("Unsupported relocation kind");
      }
    }
  }
}

/// Print atom VAs. Used only for debugging.
void AtomChunk::printAtomAddresses(uint64_t baseAddr) {
  for (const auto *layout : _atomLayouts) {
    const DefinedAtom *atom = cast<DefinedAtom>(layout->_atom);
    uint64_t addr = layout->_virtualAddr;
    llvm::dbgs() << llvm::format("0x%08llx: ", addr + baseAddr)
                 << (atom->name().empty() ? "(anonymous)" : atom->name())
                 << "\n";
  }
}

/// List all virtual addresses (and not relative virtual addresses) that need
/// to be fixed up if image base is relocated. The only relocation type that
/// needs to be fixed is DIR32 on i386. REL32 is not (and should not be)
/// fixed up because it's PC-relative.
void AtomChunk::addBaseRelocations(std::vector<uint64_t> &relocSites) {
  // TODO: llvm-objdump doesn't support parsing the base relocation table, so
  // we can't really test this at the moment. As a temporary solution, we
  // should output debug messages with atom names and addresses so that we
  // can inspect relocations, and fix the tests (base-reloc.test, maybe
  // others) to use those messages.
  for (const auto *layout : _atomLayouts) {
    const DefinedAtom *atom = cast<DefinedAtom>(layout->_atom);
    for (const Reference *ref : *atom)
      if (ref->kind() == llvm::COFF::IMAGE_REL_I386_DIR32)
        relocSites.push_back(layout->_virtualAddr + ref->offsetInAtom());
  }
}

void DataDirectoryChunk::write(uint8_t *fileBuffer) {
  if (!_atomLayouts.empty()) {
    assert(_atomLayouts.size() == 1);
    const AtomLayout *layout = _atomLayouts[0];
    ArrayRef<uint8_t> content =
        static_cast<const DefinedAtom *>(layout->_atom)->rawContent();
    std::memcpy(fileBuffer + _fileOffset, content.data(), content.size());
  }

  // Write base relocation table entry.
  int baseRelocOffset = llvm::COFF::DataDirectoryIndex::BASE_RELOCATION_TABLE *
                        sizeof(llvm::object::data_directory);
  auto *baseReloc = reinterpret_cast<llvm::object::data_directory *>(
      fileBuffer + _fileOffset + baseRelocOffset);
  baseReloc->RelativeVirtualAddress = _baseRelocAddr;
  baseReloc->Size = _baseRelocSize;
}

llvm::object::coff_section &SectionChunk::getSectionHeader() {
  if (_characteristics & llvm::COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA) {
    _sectionHeader.VirtualSize = 0;
    _sectionHeader.PointerToRawData = 0;
  } else {
    // Fix up section size before returning it. VirtualSize should be the size
    // of the actual content, and SizeOfRawData should be aligned to the section
    // alignment.
    _sectionHeader.VirtualSize = _size;
    _sectionHeader.SizeOfRawData = size();
  }
  return _sectionHeader;
}

ulittle32_t SectionChunk::getSectionCharacteristics() {
  return _sectionHeader.Characteristics;
}

void SectionChunk::appendAtom(const DefinedAtom *atom) {
  // Atom may have to be at a proper alignment boundary. If so, move the
  // pointer to make a room after the last atom before adding new one.
  _size = llvm::RoundUpToAlignment(_size, 1 << atom->alignment().powerOf2);

  // Create an AtomLayout and move the current pointer.
  auto *layout = new (_alloc) AtomLayout(atom, _size, _size);
  _atomLayouts.push_back(layout);
  _size += atom->size();
}

SectionChunk::SectionChunk(StringRef sectionName, uint32_t characteristics)
    : AtomChunk(kindSection), _characteristics(characteristics),
      _sectionHeader(createSectionHeader(sectionName, characteristics)) {
  // The section should be aligned to disk sector.
  _align = SECTOR_SIZE;
}

void SectionChunk::buildContents(const File &linkedFile,
                                 bool (*isEligible)(const DefinedAtom *)) {
  // Extract atoms from the linked file and append them to this section.
  for (const DefinedAtom *atom : linkedFile.defined()) {
    if (isEligible(atom))
      appendAtom(atom);
  }

  // Now that we have a list of atoms that to be written in this section,
  // and we know the size of the section. Let's write them to the section
  // header. VirtualSize should be the size of the actual content, and
  // SizeOfRawData should be aligned to the section alignment.
  _sectionHeader.VirtualSize = _size;
  _sectionHeader.SizeOfRawData = size();
}

llvm::object::coff_section
SectionChunk::createSectionHeader(StringRef sectionName,
                                  uint32_t characteristics) const {
  llvm::object::coff_section header;

  // Section name equal to or shorter than 8 byte fits in the section
  // header. Longer names should be stored to string table, which is not
  // implemented yet.
  if (sizeof(header.Name) < sectionName.size())
    llvm_unreachable("Cannot handle section name longer than 8 byte");

  // Name field must be NUL-padded. If the name is exactly 8 byte long,
  // there's no terminating NUL.
  std::memset(header.Name, 0, sizeof(header.Name));
  std::strncpy(header.Name, sectionName.data(),
               std::min(sizeof(header.Name), sectionName.size()));

  header.VirtualSize = 0;
  header.VirtualAddress = 0;
  header.SizeOfRawData = 0;
  header.PointerToRawData = 0;
  header.PointerToRelocations = 0;
  header.PointerToLinenumbers = 0;
  header.NumberOfRelocations = 0;
  header.NumberOfLinenumbers = 0;
  header.Characteristics = characteristics;
  return header;
}

void GenericSectionChunk::write(uint8_t *fileBuffer) {
  if (_atomLayouts.empty())
    return;
  if (_characteristics & llvm::COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA)
    return;
  if (_characteristics & llvm::COFF::IMAGE_SCN_CNT_CODE) {
    // Fill the section with INT 3 (0xCC) rather than NUL, so that the
    // disassembler will not interpret a garbage between atoms as the beginning
    // of multi-byte machine code. This does not change the behavior of
    // resulting binary but help debugging.
    uint8_t *start = fileBuffer + _atomLayouts.front()->_fileOffset;
    uint8_t *end = fileBuffer + _atomLayouts.back()->_fileOffset;
    memset(start, 0xCC, end - start);
  }
  SectionChunk::write(fileBuffer);
}

uint32_t GenericSectionChunk::getDefaultCharacteristics(
    StringRef name, const std::vector<const DefinedAtom *> &atoms) {
  const uint32_t code = llvm::COFF::IMAGE_SCN_CNT_CODE;
  const uint32_t execute = llvm::COFF::IMAGE_SCN_MEM_EXECUTE;
  const uint32_t read = llvm::COFF::IMAGE_SCN_MEM_READ;
  const uint32_t write = llvm::COFF::IMAGE_SCN_MEM_WRITE;
  const uint32_t data = llvm::COFF::IMAGE_SCN_CNT_INITIALIZED_DATA;
  const uint32_t bss = llvm::COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA;
  if (name == ".text")
    return code | execute | read;
  if (name == ".data")
    return data | read | write;
  if (name == ".rdata")
    return data | read;
  if (name == ".bss")
    return bss | read | write;
  assert(atoms.size() > 0);
  switch (atoms[0]->permissions()) {
    case DefinedAtom::permR__:
      return data | read;
    case DefinedAtom::permRW_:
      return data | read | write;
    case DefinedAtom::permR_X:
      return code | execute | read;
    case DefinedAtom::permRWX:
      return code | execute | read | write;
    default:
      llvm_unreachable("Unsupported permission");
  }
}

void SectionHeaderTableChunk::addSection(SectionChunk *chunk) {
  _sections.push_back(chunk);
}

uint64_t SectionHeaderTableChunk::size() const {
  return _sections.size() * sizeof(llvm::object::coff_section);
}

void SectionHeaderTableChunk::write(uint8_t *fileBuffer) {
  uint64_t offset = 0;
  fileBuffer += fileOffset();
  for (const auto &chunk : _sections) {
    const llvm::object::coff_section &header = chunk->getSectionHeader();
    std::memcpy(fileBuffer + offset, &header, sizeof(header));
    offset += sizeof(header);
  }
}

/// Creates .reloc section content from the other sections. The content of
/// .reloc is basically a list of relocation sites. The relocation sites are
/// divided into blocks. Each block represents the base relocation for a 4K
/// page.
///
/// By dividing 32 bit RVAs into blocks, COFF saves disk and memory space for
/// the base relocation. A block consists of a 32 bit page RVA and 16 bit
/// relocation entries which represent offsets in the page. That is a more
/// compact representation than a simple vector of 32 bit RVAs.
void BaseRelocChunk::setContents(ChunkVectorT &chunks) {
  std::vector<uint64_t> relocSites = listRelocSites(chunks);
  PageOffsetT blocks = groupByPage(relocSites);
  uint64_t ordinal = 0;
  for (auto &i : blocks) {
    uint64_t pageAddr = i.first;
    const std::vector<uint16_t> &offsetsInPage = i.second;
    appendAtom(createBaseRelocBlock(_file, ordinal++, pageAddr, offsetsInPage));
  }
}

// Returns a list of RVAs that needs to be relocated if the binary is loaded
// at an address different from its preferred one.
std::vector<uint64_t> BaseRelocChunk::listRelocSites(ChunkVectorT &chunks) {
  std::vector<uint64_t> ret;
  for (auto &cp : chunks)
    if (SectionChunk *chunk = dyn_cast<SectionChunk>(&*cp))
      chunk->addBaseRelocations(ret);
  return ret;
}

// Divide the given RVAs into blocks.
BaseRelocChunk::PageOffsetT
BaseRelocChunk::groupByPage(std::vector<uint64_t> relocSites) {
  PageOffsetT blocks;
  uint64_t mask = static_cast<uint64_t>(PAGE_SIZE) - 1;
  for (uint64_t addr : relocSites)
    blocks[addr & ~mask].push_back(addr & mask);
  return blocks;
}

// Create the content of a relocation block.
DefinedAtom *
BaseRelocChunk::createBaseRelocBlock(const File &file, uint64_t ordinal,
                                     uint64_t pageAddr,
                                     const std::vector<uint16_t> &offsets) {
  // Relocation blocks should be padded with IMAGE_REL_I386_ABSOLUTE to be
  // aligned to a DWORD size boundary.
  uint32_t size = llvm::RoundUpToAlignment(
      sizeof(ulittle32_t) * 2 + sizeof(ulittle16_t) * offsets.size(),
      sizeof(ulittle32_t));
  std::vector<uint8_t> contents(size);
  uint8_t *ptr = &contents[0];

  // The first four bytes is the page RVA.
  *reinterpret_cast<ulittle32_t *>(ptr) = pageAddr;
  ptr += sizeof(ulittle32_t);

  // The second four bytes is the size of the block, including the the page
  // RVA and this size field.
  *reinterpret_cast<ulittle32_t *>(ptr) = size;
  ptr += sizeof(ulittle32_t);

  // The rest of the block consists of offsets in the page.
  for (uint16_t offset : offsets) {
    assert(offset < PAGE_SIZE);
    uint16_t val = (llvm::COFF::IMAGE_REL_BASED_HIGHLOW << 12) | offset;
    *reinterpret_cast<ulittle16_t *>(ptr) = val;
    ptr += sizeof(ulittle16_t);
  }
  return new (_alloc) BaseRelocAtom(file, ordinal, std::move(contents));
}

} // end anonymous namespace

class ExecutableWriter : public Writer {
public:
  explicit ExecutableWriter(const PECOFFLinkingContext &context)
      : _PECOFFLinkingContext(context), _numSections(0),
        _imageSizeInMemory(PAGE_SIZE), _imageSizeOnDisk(0) {}

  void build(const File &linkedFile);
  virtual error_code writeFile(const File &linkedFile, StringRef path);

private:
  void applyAllRelocations(uint8_t *bufferStart);
  void printAllAtomAddresses();
  void addChunk(Chunk *chunk);
  void addSectionChunk(SectionChunk *chunk, SectionHeaderTableChunk *table);
  void setImageSizeOnDisk();
  void setAddressOfEntryPoint(SectionChunk *text, PEHeaderChunk *peHeader);
  uint64_t calcSectionSize(llvm::COFF::SectionCharacteristics sectionType);

  uint64_t calcSizeOfInitializedData() {
    return calcSectionSize(llvm::COFF::IMAGE_SCN_CNT_INITIALIZED_DATA);
  }

  uint64_t calcSizeOfUninitializedData() {
    return calcSectionSize(llvm::COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA);
  }

  uint64_t calcSizeOfCode() {
    return calcSectionSize(llvm::COFF::IMAGE_SCN_CNT_CODE);
  }

  std::vector<std::unique_ptr<Chunk>> _chunks;
  const PECOFFLinkingContext &_PECOFFLinkingContext;
  uint32_t _numSections;

  // The size of the image in memory. This is initialized with PAGE_SIZE, as the
  // first page starting at ImageBase is usually left unmapped. IIUC there's no
  // technical reason to do so, but we'll follow that convention so that we
  // don't produce odd-looking binary.
  uint32_t _imageSizeInMemory;

  // The size of the image on disk. This is basically the sum of all chunks in
  // the output file with paddings between them.
  uint32_t _imageSizeOnDisk;

  // The map from defined atoms to its RVAs. Will be used for relocation.
  std::map<const Atom *, uint64_t> atomRva;
};

StringRef customSectionName(const DefinedAtom *atom) {
  assert(atom->sectionChoice() == DefinedAtom::sectionCustomRequired);
  StringRef s = atom->customSectionName();
  size_t pos = s.find('$');
  return (pos == StringRef::npos) ? s : s.substr(0, pos);
}

StringRef chooseSectionByContent(const DefinedAtom *atom) {
  switch (atom->contentType()) {
  case DefinedAtom::typeCode:
    return ".text";
  case DefinedAtom::typeZeroFill:
    return ".bss";
  case DefinedAtom::typeData:
    if (atom->permissions() == DefinedAtom::permR__)
      return ".rdata";
    if (atom->permissions() == DefinedAtom::permRW_)
      return ".data";
    break;
  default:
    break;
  }
  llvm::errs() << "Atom: contentType=" << atom->contentType()
               << " permission=" << atom->permissions() << "\n";
  llvm_unreachable("Failed to choose section based on content");
}

typedef std::map<StringRef, std::vector<const DefinedAtom *> > AtomVectorMap;

void groupAtoms(const PECOFFLinkingContext &ctx, const File &file,
                AtomVectorMap &result, const DefinedAtom *&datadir) {
  for (const DefinedAtom *atom : file.defined()) {
    if (atom->sectionChoice() == DefinedAtom::sectionCustomRequired) {
      StringRef section = customSectionName(atom);
      result[ctx.getOutputSectionName(section)].push_back(atom);
      continue;
    }
    if (atom->sectionChoice() == DefinedAtom::sectionBasedOnContent) {
      if (atom->contentType() == DefinedAtom::typeDataDirectoryEntry) {
        datadir = atom;
      } else {
        StringRef section = chooseSectionByContent(atom);
        result[ctx.getOutputSectionName(section)].push_back(atom);
      }
      continue;
    }
    llvm_unreachable("Unknown section choice");
  }
}

// Create all chunks that consist of the output file.
void ExecutableWriter::build(const File &linkedFile) {
  AtomVectorMap atoms;
  const DefinedAtom *dataDirAtom = nullptr;
  groupAtoms(_PECOFFLinkingContext, linkedFile, atoms, dataDirAtom);

  // Create file chunks and add them to the list.
  auto *dosStub = new DOSStubChunk(_PECOFFLinkingContext);
  auto *peHeader = new PEHeaderChunk(_PECOFFLinkingContext);
  auto *dataDirectory = new DataDirectoryChunk(dataDirAtom);
  auto *sectionTable = new SectionHeaderTableChunk();
  addChunk(dosStub);
  addChunk(peHeader);
  addChunk(dataDirectory);
  addChunk(sectionTable);

  SectionChunk *text = nullptr;
  SectionChunk *data = nullptr;
  for (auto i : atoms) {
    StringRef sectionName = i.first;
    std::vector<const DefinedAtom *> &contents = i.second;
    auto *section = new GenericSectionChunk(_PECOFFLinkingContext, sectionName,
                                            contents);
    addSectionChunk(section, sectionTable);

    if (!text && sectionName == ".text")
      text = section;
    else if (!data && (sectionName == ".data" || sectionName == ".rdata"))
      data = section;
  }

  BaseRelocChunk *baseReloc = nullptr;
  if (_PECOFFLinkingContext.getBaseRelocationEnabled())
    baseReloc = new BaseRelocChunk(linkedFile);

  // Now that we know the addresses of all defined atoms that needs to be
  // relocated. So we can create the ".reloc" section which contains all the
  // relocation sites.
  if (baseReloc) {
    baseReloc->setContents(_chunks);
    if (baseReloc->size()) {
      addSectionChunk(baseReloc, sectionTable);
      dataDirectory->setBaseRelocField(baseReloc->getSectionRva(),
                                       baseReloc->rawSize());
    }
  }

  setImageSizeOnDisk();

  // Now that we know the size and file offset of sections. Set the file
  // header accordingly.
  peHeader->setSizeOfCode(calcSizeOfCode());
  if (text)
    peHeader->setBaseOfCode(text->getVirtualAddress());
  if (data)
    peHeader->setBaseOfData(data->getVirtualAddress());
  peHeader->setSizeOfInitializedData(calcSizeOfInitializedData());
  peHeader->setSizeOfUninitializedData(calcSizeOfUninitializedData());
  peHeader->setNumberOfSections(_numSections);
  peHeader->setSizeOfImage(_imageSizeInMemory);

  // The combined size of the DOS, PE and section headers including garbage
  // between the end of the header and the beginning of the first section.
  peHeader->setSizeOfHeaders(dosStub->size() + peHeader->size() +
                             sectionTable->size() + dataDirectory->size());

  setAddressOfEntryPoint(text, peHeader);
}

error_code ExecutableWriter::writeFile(const File &linkedFile, StringRef path) {
  this->build(linkedFile);

  uint64_t totalSize = _chunks.back()->fileOffset() + _chunks.back()->size();
  OwningPtr<llvm::FileOutputBuffer> buffer;
  error_code ec = llvm::FileOutputBuffer::create(
      path, totalSize, buffer, llvm::FileOutputBuffer::F_executable);
  if (ec)
    return ec;

  for (const auto &chunk : _chunks)
    chunk->write(buffer->getBufferStart());
  applyAllRelocations(buffer->getBufferStart());
  DEBUG(printAllAtomAddresses());
  return buffer->commit();
}

/// Apply relocations to the output file buffer. This two pass. In the first
/// pass, we visit all atoms to create a map from atom to its virtual
/// address. In the second pass, we visit all relocation references to fix
/// up addresses in the buffer.
void ExecutableWriter::applyAllRelocations(uint8_t *bufferStart) {
  for (auto &cp : _chunks)
    if (AtomChunk *chunk = dyn_cast<AtomChunk>(&*cp))
      chunk->applyRelocations(bufferStart, atomRva,
                              _PECOFFLinkingContext.getBaseAddress());
}

/// Print atom VAs. Used only for debugging.
void ExecutableWriter::printAllAtomAddresses() {
  for (auto &cp : _chunks)
    if (AtomChunk *chunk = dyn_cast<AtomChunk>(&*cp))
      chunk->printAtomAddresses(_PECOFFLinkingContext.getBaseAddress());
}

void ExecutableWriter::addChunk(Chunk *chunk) {
  _chunks.push_back(std::unique_ptr<Chunk>(chunk));
}

void ExecutableWriter::addSectionChunk(SectionChunk *chunk,
                                       SectionHeaderTableChunk *table) {
  _chunks.push_back(std::unique_ptr<Chunk>(chunk));
  table->addSection(chunk);
  _numSections++;

  // Compute and set the starting address of sections when loaded in
  // memory. They are different from positions on disk because sections need
  // to be sector-aligned on disk but page-aligned in memory.
  chunk->setVirtualAddress(_imageSizeInMemory);
  chunk->buildAtomToVirtualAddr(atomRva);
  _imageSizeInMemory =
      llvm::RoundUpToAlignment(_imageSizeInMemory + chunk->size(), PAGE_SIZE);
}

void ExecutableWriter::setImageSizeOnDisk() {
  for (auto &chunk : _chunks) {
    // Compute and set the offset of the chunk in the output file.
    _imageSizeOnDisk =
        llvm::RoundUpToAlignment(_imageSizeOnDisk, chunk->align());
    chunk->setFileOffset(_imageSizeOnDisk);
    _imageSizeOnDisk += chunk->size();
  }
}

void ExecutableWriter::setAddressOfEntryPoint(SectionChunk *text,
                                              PEHeaderChunk *peHeader) {
  // Find the virtual address of the entry point symbol if any.
  // PECOFF spec says that entry point for dll images is optional, in which
  // case it must be set to 0.
  if (_PECOFFLinkingContext.entrySymbolName().empty() &&
      _PECOFFLinkingContext.getImageType() == PECOFFLinkingContext::IMAGE_DLL) {
    peHeader->setAddressOfEntryPoint(0);
  } else {
    uint64_t entryPointAddress =
        text->getAtomVirtualAddress(_PECOFFLinkingContext.entrySymbolName());
    if (entryPointAddress != 0)
      peHeader->setAddressOfEntryPoint(entryPointAddress);
  }
}

uint64_t ExecutableWriter::calcSectionSize(
    llvm::COFF::SectionCharacteristics sectionType) {
  uint64_t ret = 0;
  for (auto &cp : _chunks)
    if (SectionChunk *chunk = dyn_cast<SectionChunk>(&*cp))
      if (chunk->getSectionCharacteristics() & sectionType)
        ret += chunk->size();
  return ret;
}

} // end namespace pecoff

std::unique_ptr<Writer> createWriterPECOFF(const PECOFFLinkingContext &info) {
  return std::unique_ptr<Writer>(new pecoff::ExecutableWriter(info));
}

} // end namespace lld
