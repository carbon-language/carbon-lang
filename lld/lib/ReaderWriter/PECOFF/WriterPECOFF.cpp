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
/// executable file.
///
/// This writer currently supports 32 bit PE/COFF for x86 processor only.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "WriterPECOFF"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <time.h>
#include <vector>

#include "Atoms.h"
#include "WriterImportLibrary.h"

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
using llvm::support::ulittle64_t;
using llvm::COFF::DataDirectoryIndex;

namespace lld {
namespace pecoff {

// Page size of x86 processor. Some data needs to be aligned at page boundary
// when loaded into memory.
static const int PAGE_SIZE = 4096;

// Disk sector size. Some data needs to be aligned at disk sector boundary in
// file.
static const int SECTOR_SIZE = 512;

namespace {
class SectionChunk;

/// A Chunk is an abstract contiguous range in an output file.
class Chunk {
public:
  enum Kind {
    kindHeader,
    kindSection,
    kindAtomChunk
  };

  explicit Chunk(Kind kind) : _kind(kind), _size(0) {}
  virtual ~Chunk() {}
  virtual void write(uint8_t *buffer) = 0;
  virtual uint64_t size() const { return _size; }
  virtual uint64_t align() const { return 1; }

  uint64_t fileOffset() const { return _fileOffset; }
  void setFileOffset(uint64_t fileOffset) { _fileOffset = fileOffset; }
  Kind getKind() const { return _kind; }

protected:
  Kind _kind;
  uint64_t _size;
  uint64_t _fileOffset;
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
  explicit DOSStubChunk(const PECOFFLinkingContext &ctx)
      : HeaderChunk(), _context(ctx) {
    // Minimum size of DOS stub is 64 bytes. The next block (PE header) needs to
    // be aligned on 8 byte boundary.
    size_t size = std::max(_context.getDosStub().size(), (size_t)64);
    _size = llvm::RoundUpToAlignment(size, 8);
  }

  virtual void write(uint8_t *buffer) {
    ArrayRef<uint8_t> array = _context.getDosStub();
    std::memcpy(buffer, array.data(), array.size());
    auto *header = reinterpret_cast<llvm::object::dos_header *>(buffer);
    header->AddressOfNewExeHeader = _size;
  }

private:
  const PECOFFLinkingContext &_context;
};

/// A PEHeaderChunk represents PE header including COFF header.
template <class PEHeader>
class PEHeaderChunk : public HeaderChunk {
public:
  explicit PEHeaderChunk(const PECOFFLinkingContext &ctx);

  virtual void write(uint8_t *buffer);

  void setSizeOfHeaders(uint64_t size) {
    // Must be multiple of FileAlignment.
    _peHeader.SizeOfHeaders = llvm::RoundUpToAlignment(size, SECTOR_SIZE);
  }

  void setSizeOfCode(uint64_t size) { _peHeader.SizeOfCode = size; }
  void setBaseOfCode(uint32_t rva) { _peHeader.BaseOfCode = rva; }
  void setBaseOfData(uint32_t rva);
  void setSizeOfImage(uint32_t size) { _peHeader.SizeOfImage = size; }

  void setSizeOfInitializedData(uint64_t size) {
    _peHeader.SizeOfInitializedData = size;
  }

  void setSizeOfUninitializedData(uint64_t size) {
    _peHeader.SizeOfUninitializedData = size;
  }

  void setNumberOfSections(uint32_t num) { _coffHeader.NumberOfSections = num; }

  void setAddressOfEntryPoint(uint32_t address) {
    _peHeader.AddressOfEntryPoint = address;
  }

private:
  llvm::object::coff_file_header _coffHeader;
  PEHeader _peHeader;
};

/// A SectionHeaderTableChunk represents Section Table Header of PE/COFF
/// format, which is a list of section headers.
class SectionHeaderTableChunk : public HeaderChunk {
public:
  SectionHeaderTableChunk() : HeaderChunk() {}
  void addSection(SectionChunk *chunk);
  virtual uint64_t size() const;
  virtual void write(uint8_t *buffer);

private:
  static llvm::object::coff_section createSectionHeader(SectionChunk *chunk);

  std::vector<SectionChunk *> _sections;
};

class SectionChunk : public Chunk {
public:
  virtual uint64_t align() const { return SECTOR_SIZE; }
  uint32_t getCharacteristics() const { return _characteristics; }
  StringRef getSectionName() const { return _sectionName; }

  static bool classof(const Chunk *c) {
    Kind kind = c->getKind();
    return kind == kindSection || kind == kindAtomChunk;
  }

  uint64_t getVirtualAddress() { return _virtualAddress; }
  virtual void setVirtualAddress(uint32_t rva) { _virtualAddress = rva; }

protected:
  SectionChunk(Kind kind, StringRef sectionName, uint32_t characteristics)
      : Chunk(kind), _sectionName(sectionName),
        _characteristics(characteristics), _virtualAddress(0) {}

private:
  StringRef _sectionName;
  const uint32_t _characteristics;
  uint64_t _virtualAddress;
};

/// An AtomChunk represents a section containing atoms.
class AtomChunk : public SectionChunk {
public:
  AtomChunk(const PECOFFLinkingContext &ctx, StringRef name,
            const std::vector<const DefinedAtom *> &atoms);

  virtual void write(uint8_t *buffer);

  void appendAtom(const DefinedAtom *atom);
  void buildAtomRvaMap(std::map<const Atom *, uint64_t> &atomRva) const;

  void applyRelocations32(uint8_t *buffer,
                          std::map<const Atom *, uint64_t> &atomRva,
                          std::vector<uint64_t> &sectionRva,
                          uint64_t imageBaseAddress);
  void applyRelocations64(uint8_t *buffer,
                          std::map<const Atom *, uint64_t> &atomRva,
                          std::vector<uint64_t> &sectionRva,
                          uint64_t imageBaseAddress);

  void printAtomAddresses(uint64_t baseAddr) const;
  void addBaseRelocations(std::vector<uint64_t> &relocSites) const;

  virtual void setVirtualAddress(uint32_t rva);
  uint64_t getAtomVirtualAddress(StringRef name) const;

  static bool classof(const Chunk *c) { return c->getKind() == kindAtomChunk; }

protected:
  std::vector<AtomLayout *> _atomLayouts;
  uint64_t _virtualAddress;

private:
  uint32_t
  computeCharacteristics(const PECOFFLinkingContext &ctx, StringRef name,
                         const std::vector<const DefinedAtom *> &atoms) const {
    return ctx.getSectionAttributes(name,
                                    getDefaultCharacteristics(name, atoms));
  }

  uint32_t getDefaultCharacteristics(
      StringRef name, const std::vector<const DefinedAtom *> &atoms) const;

  mutable llvm::BumpPtrAllocator _alloc;
};

/// A DataDirectoryChunk represents data directory entries that follows the PE
/// header in the output file. An entry consists of an 8 byte field that
/// indicates a relative virtual address (the starting address of the entry data
/// in memory) and 8 byte entry data size.
class DataDirectoryChunk : public HeaderChunk {
public:
  DataDirectoryChunk()
      : HeaderChunk(), _data(std::vector<llvm::object::data_directory>(16)) {}

  virtual uint64_t size() const {
    return sizeof(llvm::object::data_directory) * _data.size();
  }

  void setField(DataDirectoryIndex index, uint32_t addr, uint32_t size);
  virtual void write(uint8_t *buffer);

private:
  std::vector<llvm::object::data_directory> _data;
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
  typedef std::vector<std::unique_ptr<Chunk> > ChunkVectorT;
  typedef std::map<uint64_t, std::vector<uint16_t> > PageOffsetT;

public:
  BaseRelocChunk(ChunkVectorT &chunks)
      : SectionChunk(kindSection, ".reloc", characteristics),
        _contents(createContents(chunks)) {}

  virtual void write(uint8_t *buffer) {
    std::memcpy(buffer, &_contents[0], _contents.size());
  }

  virtual uint64_t size() const { return _contents.size(); }

private:
  // When loaded into memory, reloc section should be readable and writable.
  static const uint32_t characteristics =
      llvm::COFF::IMAGE_SCN_MEM_READ |
      llvm::COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
      llvm::COFF::IMAGE_SCN_MEM_DISCARDABLE;

  std::vector<uint8_t> createContents(ChunkVectorT &chunks) const;

  // Returns a list of RVAs that needs to be relocated if the binary is loaded
  // at an address different from its preferred one.
  std::vector<uint64_t> listRelocSites(ChunkVectorT &chunks) const;

  // Divide the given RVAs into blocks.
  PageOffsetT groupByPage(const std::vector<uint64_t> &relocSites) const;

  // Create the content of a relocation block.
  std::vector<uint8_t>
  createBaseRelocBlock(uint64_t pageAddr,
                       const std::vector<uint16_t> &offsets) const;

  std::vector<uint8_t> _contents;
};

template <class PEHeader>
PEHeaderChunk<PEHeader>::PEHeaderChunk(const PECOFFLinkingContext &ctx)
    : HeaderChunk() {
  // Set the size of the chunk and initialize the header with null bytes.
  _size = sizeof(llvm::COFF::PEMagic) + sizeof(_coffHeader) + sizeof(_peHeader);
  std::memset(&_coffHeader, 0, sizeof(_coffHeader));
  std::memset(&_peHeader, 0, sizeof(_peHeader));

  _coffHeader.Machine = ctx.getMachineType();
  _coffHeader.TimeDateStamp = time(nullptr);

  // Attributes of the executable.
  uint16_t characteristics = llvm::COFF::IMAGE_FILE_EXECUTABLE_IMAGE;
  if (!ctx.is64Bit())
    characteristics |= llvm::COFF::IMAGE_FILE_32BIT_MACHINE;
  if (ctx.getLargeAddressAware() || ctx.is64Bit())
    characteristics |= llvm::COFF::IMAGE_FILE_LARGE_ADDRESS_AWARE;
  if (ctx.getSwapRunFromCD())
    characteristics |= llvm::COFF::IMAGE_FILE_REMOVABLE_RUN_FROM_SWAP;
  if (ctx.getSwapRunFromNet())
    characteristics |= llvm::COFF::IMAGE_FILE_NET_RUN_FROM_SWAP;
  if (!ctx.getBaseRelocationEnabled())
    characteristics |= llvm::COFF::IMAGE_FILE_RELOCS_STRIPPED;

  _coffHeader.Characteristics = characteristics;

  _peHeader.Magic = ctx.is64Bit() ? llvm::COFF::PE32Header::PE32_PLUS
                                  : llvm::COFF::PE32Header::PE32;

  // The address of entry point relative to ImageBase. Windows executable
  // usually starts at address 0x401000.
  _peHeader.AddressOfEntryPoint = 0x1000;

  // The address of the executable when loaded into memory. The default for
  // DLLs is 0x10000000. The default for executables is 0x400000.
  _peHeader.ImageBase = ctx.getBaseAddress();

  // Sections should be page-aligned when loaded into memory, which is 4KB on
  // x86.
  _peHeader.SectionAlignment = ctx.getSectionDefaultAlignment();

  // Sections in an executable file on disk should be sector-aligned (512 byte).
  _peHeader.FileAlignment = SECTOR_SIZE;

  // The version number of the resultant executable/DLL. The number is purely
  // informative, and neither the linker nor the loader won't use it. User can
  // set the value using /version command line option. Default is 0.0.
  PECOFFLinkingContext::Version imageVersion = ctx.getImageVersion();
  _peHeader.MajorImageVersion = imageVersion.majorVersion;
  _peHeader.MinorImageVersion = imageVersion.minorVersion;

  // The required Windows version number. This is the internal version and
  // shouldn't be confused with product name. Windows 7 is version 6.1 and
  // Windows 8 is 6.2, for example.
  PECOFFLinkingContext::Version minOSVersion = ctx.getMinOSVersion();
  _peHeader.MajorOperatingSystemVersion = minOSVersion.majorVersion;
  _peHeader.MinorOperatingSystemVersion = minOSVersion.minorVersion;
  _peHeader.MajorSubsystemVersion = minOSVersion.majorVersion;
  _peHeader.MinorSubsystemVersion = minOSVersion.minorVersion;

  _peHeader.Subsystem = ctx.getSubsystem();

  // Despite its name, DLL characteristics field has meaning both for
  // executables and DLLs. We are not very sure if the following bits must
  // be set, but regular binaries seem to have these bits, so we follow
  // them.
  uint16_t dllCharacteristics = 0;
  if (ctx.noSEH())
    dllCharacteristics |= llvm::COFF::IMAGE_DLL_CHARACTERISTICS_NO_SEH;
  if (ctx.isTerminalServerAware())
    dllCharacteristics |=
        llvm::COFF::IMAGE_DLL_CHARACTERISTICS_TERMINAL_SERVER_AWARE;
  if (ctx.isNxCompat())
    dllCharacteristics |= llvm::COFF::IMAGE_DLL_CHARACTERISTICS_NX_COMPAT;
  if (ctx.getDynamicBaseEnabled())
    dllCharacteristics |= llvm::COFF::IMAGE_DLL_CHARACTERISTICS_DYNAMIC_BASE;
  if (!ctx.getAllowBind())
    dllCharacteristics |= llvm::COFF::IMAGE_DLL_CHARACTERISTICS_NO_BIND;
  if (!ctx.getAllowIsolation())
    dllCharacteristics |= llvm::COFF::IMAGE_DLL_CHARACTERISTICS_NO_ISOLATION;
  _peHeader.DLLCharacteristics = dllCharacteristics;

  _peHeader.SizeOfStackReserve = ctx.getStackReserve();
  _peHeader.SizeOfStackCommit = ctx.getStackCommit();
  _peHeader.SizeOfHeapReserve = ctx.getHeapReserve();
  _peHeader.SizeOfHeapCommit = ctx.getHeapCommit();

  // The number of data directory entries. We always have 16 entries.
  _peHeader.NumberOfRvaAndSize = 16;

  // The size of PE header including optional data directory.
  _coffHeader.SizeOfOptionalHeader = sizeof(PEHeader) +
      _peHeader.NumberOfRvaAndSize * sizeof(llvm::object::data_directory);
}

template <>
void PEHeaderChunk<llvm::object::pe32_header>::setBaseOfData(uint32_t rva) {
  _peHeader.BaseOfData = rva;
}

template <>
void PEHeaderChunk<llvm::object::pe32plus_header>::setBaseOfData(uint32_t rva) {
  // BaseOfData field does not exist in PE32+ header.
}

template <class PEHeader>
void PEHeaderChunk<PEHeader>::write(uint8_t *buffer) {
  std::memcpy(buffer, llvm::COFF::PEMagic, sizeof(llvm::COFF::PEMagic));
  buffer += sizeof(llvm::COFF::PEMagic);
  std::memcpy(buffer, &_coffHeader, sizeof(_coffHeader));
  buffer += sizeof(_coffHeader);
  std::memcpy(buffer, &_peHeader, sizeof(_peHeader));
}

AtomChunk::AtomChunk(const PECOFFLinkingContext &ctx, StringRef sectionName,
                     const std::vector<const DefinedAtom *> &atoms)
    : SectionChunk(kindAtomChunk, sectionName,
                   computeCharacteristics(ctx, sectionName, atoms)),
      _virtualAddress(0) {
  for (auto *a : atoms)
    appendAtom(a);
}

void AtomChunk::write(uint8_t *buffer) {
  if (_atomLayouts.empty())
    return;
  if (getCharacteristics() & llvm::COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA)
    return;
  if (getCharacteristics() & llvm::COFF::IMAGE_SCN_CNT_CODE) {
    // Fill the section with INT 3 (0xCC) rather than NUL, so that the
    // disassembler will not interpret a garbage between atoms as the beginning
    // of multi-byte machine code. This does not change the behavior of
    // resulting binary but help debugging.
    uint8_t *start = buffer + _atomLayouts.front()->_fileOffset;
    uint8_t *end = buffer + _atomLayouts.back()->_fileOffset;
    memset(start, 0xCC, end - start);
  }

  for (const auto *layout : _atomLayouts) {
    const DefinedAtom *atom = cast<DefinedAtom>(layout->_atom);
    ArrayRef<uint8_t> rawContent = atom->rawContent();
    std::memcpy(buffer + layout->_fileOffset, rawContent.data(),
                rawContent.size());
  }
}

// Add all atoms to the given map. This data will be used to do relocation.
void
AtomChunk::buildAtomRvaMap(std::map<const Atom *, uint64_t> &atomRva) const {
  for (const auto *layout : _atomLayouts)
    atomRva[layout->_atom] = layout->_virtualAddr;
}

static int getSectionIndex(uint64_t targetAddr,
                           const std::vector<uint64_t> &sectionRva) {
  int i = 1;
  for (uint64_t rva : sectionRva) {
    if (targetAddr < rva)
      return i;
    ++i;
  }
  return i;
}

static uint32_t getSectionStartAddr(uint64_t targetAddr,
                                    const std::vector<uint64_t> &sectionRva) {
  for (int i = 0, e = sectionRva.size(); i < e; ++i)
    if (i == e - 1 || (sectionRva[i] <= targetAddr && targetAddr <= sectionRva[i + 1]))
      return sectionRva[i];
  llvm_unreachable("Section missing");
}

void AtomChunk::applyRelocations32(uint8_t *buffer,
                                   std::map<const Atom *, uint64_t> &atomRva,
                                   std::vector<uint64_t> &sectionRva,
                                   uint64_t imageBaseAddress) {
  buffer += _fileOffset;
  for (const auto *layout : _atomLayouts) {
    const DefinedAtom *atom = cast<DefinedAtom>(layout->_atom);
    for (const Reference *ref : *atom) {
      // Skip if this reference is not for COFF relocation.
      if (ref->kindNamespace() != Reference::KindNamespace::COFF)
        continue;
      auto relocSite32 = reinterpret_cast<ulittle32_t *>(
          buffer + layout->_fileOffset + ref->offsetInAtom());
      auto relocSite16 = reinterpret_cast<ulittle16_t *>(relocSite32);
      uint64_t targetAddr = atomRva[ref->target()];
      // Also account for whatever offset is already stored at the relocation
      // site.
      targetAddr += *relocSite32;
      switch (ref->kindValue()) {
      case llvm::COFF::IMAGE_REL_I386_ABSOLUTE:
        // This relocation is no-op.
        break;
      case llvm::COFF::IMAGE_REL_I386_DIR32:
        // Set target's 32-bit VA.
        *relocSite32 = targetAddr + imageBaseAddress;
        break;
      case llvm::COFF::IMAGE_REL_I386_DIR32NB:
        // Set target's 32-bit RVA.
        *relocSite32 = targetAddr;
        break;
      case llvm::COFF::IMAGE_REL_I386_REL32: {
        // Set 32-bit relative address of the target. This relocation is
        // usually used for relative branch or call instruction.
        uint32_t disp = atomRva[atom] + ref->offsetInAtom() + 4;
        *relocSite32 = targetAddr - disp;
        break;
      }
      case llvm::COFF::IMAGE_REL_I386_SECTION:
        // The 16-bit section index that contains the target symbol.
        *relocSite16 = getSectionIndex(targetAddr, sectionRva);
        break;
      case llvm::COFF::IMAGE_REL_I386_SECREL:
        // The 32-bit relative address from the beginning of the section that
        // contains the target symbol.
        *relocSite32 = targetAddr - getSectionStartAddr(targetAddr, sectionRva);
        break;
      default:
        llvm_unreachable("Unsupported relocation kind");
      }
    }
  }
}

void AtomChunk::applyRelocations64(uint8_t *buffer,
                                   std::map<const Atom *, uint64_t> &atomRva,
                                   std::vector<uint64_t> &sectionRva,
                                   uint64_t imageBase) {
  buffer += _fileOffset;
  for (const auto *layout : _atomLayouts) {
    const DefinedAtom *atom = cast<DefinedAtom>(layout->_atom);
    for (const Reference *ref : *atom) {
      if (ref->kindNamespace() != Reference::KindNamespace::COFF)
        continue;

      uint8_t *loc = buffer + layout->_fileOffset + ref->offsetInAtom();
      auto relocSite16 = reinterpret_cast<ulittle16_t *>(loc);
      auto relocSite32 = reinterpret_cast<ulittle32_t *>(loc);
      auto relocSite64 = reinterpret_cast<ulittle64_t *>(loc);
      uint64_t targetAddr = atomRva[ref->target()];

      switch (ref->kindValue()) {
      case llvm::COFF::IMAGE_REL_AMD64_ADDR64:
        *relocSite64 = targetAddr;
        break;
      case llvm::COFF::IMAGE_REL_AMD64_ADDR32:
        *relocSite32 = targetAddr;
        break;
      case llvm::COFF::IMAGE_REL_AMD64_ADDR32NB:
        *relocSite32 = targetAddr - imageBase;
        break;
      case llvm::COFF::IMAGE_REL_AMD64_REL32:
        *relocSite32 = targetAddr - atomRva[atom] + ref->offsetInAtom() + 4;
        break;

#define REL32(x)                                                             \
      case llvm::COFF::IMAGE_REL_AMD64_REL32_ ## x: {                        \
        uint32_t off = targetAddr - atomRva[atom] + ref->offsetInAtom() + 4; \
        *relocSite32 = off + x;                                              \
      }
      REL32(1);
      REL32(2);
      REL32(3);
      REL32(4);
      REL32(5);
#undef CASE

      case llvm::COFF::IMAGE_REL_AMD64_SECTION:
        *relocSite16 = getSectionIndex(targetAddr, sectionRva);
        break;
      case llvm::COFF::IMAGE_REL_AMD64_SECREL:
        *relocSite32 = targetAddr - getSectionStartAddr(targetAddr, sectionRva);
        break;

      default:
        llvm::errs() << "Kind: " << (int)ref->kindValue() << "\n";
        llvm_unreachable("Unsupported relocation kind");
      }
    }
  }
}

/// Print atom VAs. Used only for debugging.
void AtomChunk::printAtomAddresses(uint64_t baseAddr) const {
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
void AtomChunk::addBaseRelocations(std::vector<uint64_t> &relocSites) const {
  // TODO: llvm-objdump doesn't support parsing the base relocation table, so
  // we can't really test this at the moment. As a temporary solution, we
  // should output debug messages with atom names and addresses so that we
  // can inspect relocations, and fix the tests (base-reloc.test, maybe
  // others) to use those messages.
  for (const auto *layout : _atomLayouts) {
    const DefinedAtom *atom = cast<DefinedAtom>(layout->_atom);
    for (const Reference *ref : *atom)
      if ((ref->kindNamespace() == Reference::KindNamespace::COFF) &&
          (ref->kindValue() == llvm::COFF::IMAGE_REL_I386_DIR32))
        relocSites.push_back(layout->_virtualAddr + ref->offsetInAtom());
  }
}

void AtomChunk::setVirtualAddress(uint32_t rva) {
  SectionChunk::setVirtualAddress(rva);
  for (AtomLayout *layout : _atomLayouts)
    layout->_virtualAddr += rva;
}

uint64_t AtomChunk::getAtomVirtualAddress(StringRef name) const {
  for (auto atomLayout : _atomLayouts)
    if (atomLayout->_atom->name() == name)
      return atomLayout->_virtualAddr;
  return 0;
}

void DataDirectoryChunk::setField(DataDirectoryIndex index, uint32_t addr,
                                  uint32_t size) {
  llvm::object::data_directory &dir = _data[index];
  dir.RelativeVirtualAddress = addr;
  dir.Size = size;
}

void DataDirectoryChunk::write(uint8_t *buffer) {
  std::memcpy(buffer, &_data[0], size());
}

void AtomChunk::appendAtom(const DefinedAtom *atom) {
  // Atom may have to be at a proper alignment boundary. If so, move the
  // pointer to make a room after the last atom before adding new one.
  _size = llvm::RoundUpToAlignment(_size, 1 << atom->alignment().powerOf2);

  // Create an AtomLayout and move the current pointer.
  auto *layout = new (_alloc) AtomLayout(atom, _size, _size);
  _atomLayouts.push_back(layout);
  _size += atom->size();
}

uint32_t AtomChunk::getDefaultCharacteristics(
    StringRef name, const std::vector<const DefinedAtom *> &atoms) const {
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

void SectionHeaderTableChunk::write(uint8_t *buffer) {
  uint64_t offset = 0;
  for (SectionChunk *chunk : _sections) {
    llvm::object::coff_section header = createSectionHeader(chunk);
    std::memcpy(buffer + offset, &header, sizeof(header));
    offset += sizeof(header);
  }
}

llvm::object::coff_section
SectionHeaderTableChunk::createSectionHeader(SectionChunk *chunk) {
  llvm::object::coff_section header;

  // Section name must be equal to or less than 8 characters in the
  // executable. Longer names will be truncated.
  StringRef sectionName = chunk->getSectionName();

  // Name field must be NUL-padded. If the name is exactly 8 byte long,
  // there's no terminating NUL.
  std::memset(header.Name, 0, sizeof(header.Name));
  std::strncpy(header.Name, sectionName.data(),
               std::min(sizeof(header.Name), sectionName.size()));

  uint32_t characteristics = chunk->getCharacteristics();
  header.VirtualAddress = chunk->getVirtualAddress();
  header.PointerToRelocations = 0;
  header.PointerToLinenumbers = 0;
  header.NumberOfRelocations = 0;
  header.NumberOfLinenumbers = 0;
  header.SizeOfRawData = chunk->size();
  header.Characteristics = characteristics;

  if (characteristics & llvm::COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA) {
    header.VirtualSize = 0;
    header.PointerToRawData = 0;
  } else {
    header.VirtualSize = chunk->size();
    header.PointerToRawData = chunk->fileOffset();
  }
  return header;
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
std::vector<uint8_t>
BaseRelocChunk::createContents(ChunkVectorT &chunks) const {
  std::vector<uint8_t> contents;
  std::vector<uint64_t> relocSites = listRelocSites(chunks);
  PageOffsetT blocks = groupByPage(relocSites);
  for (auto &i : blocks) {
    uint64_t pageAddr = i.first;
    const std::vector<uint16_t> &offsetsInPage = i.second;
    std::vector<uint8_t> block = createBaseRelocBlock(pageAddr, offsetsInPage);
    contents.insert(contents.end(), block.begin(), block.end());
  }
  return contents;
}

// Returns a list of RVAs that needs to be relocated if the binary is loaded
// at an address different from its preferred one.
std::vector<uint64_t>
BaseRelocChunk::listRelocSites(ChunkVectorT &chunks) const {
  std::vector<uint64_t> ret;
  for (auto &cp : chunks)
    if (AtomChunk *chunk = dyn_cast<AtomChunk>(&*cp))
      chunk->addBaseRelocations(ret);
  return ret;
}

// Divide the given RVAs into blocks.
BaseRelocChunk::PageOffsetT
BaseRelocChunk::groupByPage(const std::vector<uint64_t> &relocSites) const {
  PageOffsetT blocks;
  uint64_t mask = static_cast<uint64_t>(PAGE_SIZE) - 1;
  for (uint64_t addr : relocSites)
    blocks[addr & ~mask].push_back(addr & mask);
  return blocks;
}

// Create the content of a relocation block.
std::vector<uint8_t> BaseRelocChunk::createBaseRelocBlock(
    uint64_t pageAddr, const std::vector<uint16_t> &offsets) const {
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
  return contents;
}

} // end anonymous namespace

class PECOFFWriter : public Writer {
public:
  explicit PECOFFWriter(const PECOFFLinkingContext &context)
      : _ctx(context), _numSections(0), _imageSizeInMemory(PAGE_SIZE),
        _imageSizeOnDisk(0) {}

  template <class PEHeader> void build(const File &linkedFile);
  virtual error_code writeFile(const File &linkedFile, StringRef path);

private:
  void applyAllRelocations(uint8_t *bufferStart);
  void printAllAtomAddresses() const;
  void addChunk(Chunk *chunk);
  void addSectionChunk(SectionChunk *chunk, SectionHeaderTableChunk *table);
  void setImageSizeOnDisk();
  uint64_t
  calcSectionSize(llvm::COFF::SectionCharacteristics sectionType) const;

  uint64_t calcSizeOfInitializedData() const {
    return calcSectionSize(llvm::COFF::IMAGE_SCN_CNT_INITIALIZED_DATA);
  }

  uint64_t calcSizeOfUninitializedData() const {
    return calcSectionSize(llvm::COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA);
  }

  uint64_t calcSizeOfCode() const {
    return calcSectionSize(llvm::COFF::IMAGE_SCN_CNT_CODE);
  }

  std::vector<std::unique_ptr<Chunk> > _chunks;
  const PECOFFLinkingContext &_ctx;
  uint32_t _numSections;

  // The size of the image in memory. This is initialized with PAGE_SIZE, as the
  // first page starting at ImageBase is usually left unmapped. IIUC there's no
  // technical reason to do so, but we'll follow that convention so that we
  // don't produce odd-looking binary.
  uint32_t _imageSizeInMemory;

  // The size of the image on disk. This is basically the sum of all chunks in
  // the output file with paddings between them.
  uint32_t _imageSizeOnDisk;
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
                AtomVectorMap &result) {
  for (const DefinedAtom *atom : file.defined()) {
    if (atom->sectionChoice() == DefinedAtom::sectionCustomRequired) {
      StringRef section = customSectionName(atom);
      result[ctx.getOutputSectionName(section)].push_back(atom);
      continue;
    }
    if (atom->sectionChoice() == DefinedAtom::sectionBasedOnContent) {
      StringRef section = chooseSectionByContent(atom);
      result[ctx.getOutputSectionName(section)].push_back(atom);
      continue;
    }
    llvm_unreachable("Unknown section choice");
  }
}

// Create all chunks that consist of the output file.
template <class PEHeader>
void PECOFFWriter::build(const File &linkedFile) {
  AtomVectorMap atoms;
  groupAtoms(_ctx, linkedFile, atoms);

  // Create file chunks and add them to the list.
  auto *dosStub = new DOSStubChunk(_ctx);
  auto *peHeader = new PEHeaderChunk<PEHeader>(_ctx);
  auto *dataDirectory = new DataDirectoryChunk();
  auto *sectionTable = new SectionHeaderTableChunk();
  addChunk(dosStub);
  addChunk(peHeader);
  addChunk(dataDirectory);
  addChunk(sectionTable);

  for (auto i : atoms) {
    StringRef sectionName = i.first;
    std::vector<const DefinedAtom *> &contents = i.second;
    auto *section = new AtomChunk(_ctx, sectionName, contents);
    if (section->size() > 0)
      addSectionChunk(section, sectionTable);
  }

  // Now that we know the addresses of all defined atoms that needs to be
  // relocated. So we can create the ".reloc" section which contains all the
  // relocation sites.
  if (_ctx.getBaseRelocationEnabled()) {
    BaseRelocChunk *baseReloc = new BaseRelocChunk(_chunks);
    if (baseReloc->size()) {
      addSectionChunk(baseReloc, sectionTable);
      dataDirectory->setField(DataDirectoryIndex::BASE_RELOCATION_TABLE,
                              baseReloc->getVirtualAddress(),
                              baseReloc->size());
    }
  }

  setImageSizeOnDisk();

  for (std::unique_ptr<Chunk> &chunk : _chunks) {
    SectionChunk *section = dyn_cast<SectionChunk>(chunk.get());
    if (!section)
      continue;
    if (section->getSectionName() == ".text") {
      peHeader->setBaseOfCode(section->getVirtualAddress());

      // Find the virtual address of the entry point symbol if any.  PECOFF spec
      // says that entry point for dll images is optional, in which case it must
      // be set to 0.
      if (_ctx.entrySymbolName().empty() && _ctx.isDll()) {
        peHeader->setAddressOfEntryPoint(0);
      } else {
        uint64_t entryPointAddress =
            dyn_cast<AtomChunk>(section)
                ->getAtomVirtualAddress(_ctx.entrySymbolName());
        if (entryPointAddress != 0)
          peHeader->setAddressOfEntryPoint(entryPointAddress);
      }
    }
    if (section->getSectionName() == ".data")
      peHeader->setBaseOfData(section->getVirtualAddress());
    if (section->getSectionName() == ".pdata")
      dataDirectory->setField(DataDirectoryIndex::EXCEPTION_TABLE,
                              section->getVirtualAddress(), section->size());
    if (section->getSectionName() == ".idata.a")
      dataDirectory->setField(DataDirectoryIndex::IAT,
                              section->getVirtualAddress(), section->size());
    if (section->getSectionName() == ".idata.d")
      dataDirectory->setField(DataDirectoryIndex::IMPORT_TABLE,
                              section->getVirtualAddress(), section->size());
    if (section->getSectionName() == ".edata")
      dataDirectory->setField(DataDirectoryIndex::EXPORT_TABLE,
                              section->getVirtualAddress(), section->size());
    if (section->getSectionName() == ".loadcfg")
      dataDirectory->setField(DataDirectoryIndex::LOAD_CONFIG_TABLE,
                              section->getVirtualAddress(), section->size());
  }

  // Now that we know the size and file offset of sections. Set the file
  // header accordingly.
  peHeader->setSizeOfCode(calcSizeOfCode());
  peHeader->setSizeOfInitializedData(calcSizeOfInitializedData());
  peHeader->setSizeOfUninitializedData(calcSizeOfUninitializedData());
  peHeader->setNumberOfSections(_numSections);
  peHeader->setSizeOfImage(_imageSizeInMemory);
  peHeader->setSizeOfHeaders(sectionTable->fileOffset() + sectionTable->size());
}

error_code PECOFFWriter::writeFile(const File &linkedFile, StringRef path) {
  if (_ctx.is64Bit()) {
    this->build<llvm::object::pe32plus_header>(linkedFile);
  } else {
    this->build<llvm::object::pe32_header>(linkedFile);
  }

  uint64_t totalSize = _chunks.back()->fileOffset() + _chunks.back()->size();
  OwningPtr<llvm::FileOutputBuffer> buffer;
  error_code ec = llvm::FileOutputBuffer::create(
      path, totalSize, buffer, llvm::FileOutputBuffer::F_executable);
  if (ec)
    return ec;

  for (std::unique_ptr<Chunk> &chunk : _chunks)
    chunk->write(buffer->getBufferStart() + chunk->fileOffset());
  applyAllRelocations(buffer->getBufferStart());
  DEBUG(printAllAtomAddresses());

  if (_ctx.isDll())
    writeImportLibrary(_ctx);

  return buffer->commit();
}

/// Apply relocations to the output file buffer. This two pass. In the first
/// pass, we visit all atoms to create a map from atom to its virtual
/// address. In the second pass, we visit all relocation references to fix
/// up addresses in the buffer.
void PECOFFWriter::applyAllRelocations(uint8_t *bufferStart) {
  std::map<const Atom *, uint64_t> atomRva;
  std::vector<uint64_t> sectionRva;

  // Create the list of section start addresses.
  for (auto &cp : _chunks)
    if (SectionChunk *section = dyn_cast<SectionChunk>(&*cp))
      sectionRva.push_back(section->getVirtualAddress());

  // Pass 1
  for (auto &cp : _chunks)
    if (AtomChunk *chunk = dyn_cast<AtomChunk>(&*cp))
      chunk->buildAtomRvaMap(atomRva);

  // Pass 2
  uint64_t base = _ctx.getBaseAddress();
  for (auto &cp : _chunks) {
    if (AtomChunk *chunk = dyn_cast<AtomChunk>(&*cp)) {
      if (_ctx.is64Bit()) {
        chunk->applyRelocations64(bufferStart, atomRva, sectionRva, base);
      } else {
        chunk->applyRelocations32(bufferStart, atomRva, sectionRva, base);
      }
    }
  }
}

/// Print atom VAs. Used only for debugging.
void PECOFFWriter::printAllAtomAddresses() const {
  for (auto &cp : _chunks)
    if (AtomChunk *chunk = dyn_cast<AtomChunk>(&*cp))
      chunk->printAtomAddresses(_ctx.getBaseAddress());
}

void PECOFFWriter::addChunk(Chunk *chunk) {
  _chunks.push_back(std::unique_ptr<Chunk>(chunk));
}

void PECOFFWriter::addSectionChunk(SectionChunk *chunk,
                                   SectionHeaderTableChunk *table) {
  _chunks.push_back(std::unique_ptr<Chunk>(chunk));
  table->addSection(chunk);
  _numSections++;

  // Compute and set the starting address of sections when loaded in
  // memory. They are different from positions on disk because sections need
  // to be sector-aligned on disk but page-aligned in memory.
  chunk->setVirtualAddress(_imageSizeInMemory);
  _imageSizeInMemory =
      llvm::RoundUpToAlignment(_imageSizeInMemory + chunk->size(), PAGE_SIZE);
}

void PECOFFWriter::setImageSizeOnDisk() {
  for (auto &chunk : _chunks) {
    // Compute and set the offset of the chunk in the output file.
    _imageSizeOnDisk =
        llvm::RoundUpToAlignment(_imageSizeOnDisk, chunk->align());
    chunk->setFileOffset(_imageSizeOnDisk);
    _imageSizeOnDisk += chunk->size();
  }
}

uint64_t PECOFFWriter::calcSectionSize(
    llvm::COFF::SectionCharacteristics sectionType) const {
  uint64_t ret = 0;
  for (auto &cp : _chunks)
    if (SectionChunk *chunk = dyn_cast<SectionChunk>(&*cp))
      if (chunk->getCharacteristics() & sectionType)
        ret += chunk->size();
  return ret;
}

} // end namespace pecoff

std::unique_ptr<Writer> createWriterPECOFF(const PECOFFLinkingContext &info) {
  return std::unique_ptr<Writer>(new pecoff::PECOFFWriter(info));
}

} // end namespace lld
