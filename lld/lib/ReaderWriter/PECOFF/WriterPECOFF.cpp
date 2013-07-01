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
/// This writer is reponsible for writing Core Linker results to an Windows
/// executable file. Currently it can only output ".text" section; other
/// sections including the symbol table are silently ignored.
///
/// This writer currently supports 32 bit PE/COFF for x86 processor only.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "WriterPECOFF"

#include <map>
#include <time.h>
#include <vector>

#include "lld/Core/DefinedAtom.h"
#include "lld/Core/File.h"
#include "lld/Core/InputFiles.h"
#include "lld/ReaderWriter/AtomLayout.h"
#include "lld/ReaderWriter/PECOFFTargetInfo.h"
#include "lld/ReaderWriter/Writer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileOutputBuffer.h"

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

// The address of the executable when loaded into memory.
const int32_t IMAGE_BASE = 0x400000;

/// A Chunk is an abstrace contiguous range in an output file.
class Chunk {
public:
  enum Kind {
    kindHeader,
    kindSection
  };

  Chunk(Kind kind) : _kind(kind), _size(0), _align(1) {}
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
  DOSStubChunk() : HeaderChunk() {
    // Make the DOS stub occupy the first 128 bytes of an exe. Technically
    // this can be as small as 64 bytes, but GNU binutil's objdump cannot
    // parse such irregular header.
    _size = 128;

    // A DOS stub is usually a small valid DOS program that prints out a message
    // "This program requires Microsoft Windows" to help user who accidentally
    // run a Windows executable on DOS. That's not a technical requirement, so
    // we don't bother to emit such code, at least for now. We simply fill the
    // DOS stub with null bytes.
    std::memset(&_dosHeader, 0, sizeof(_dosHeader));

    _dosHeader.Magic = 'M' | ('Z' << 8);
    _dosHeader.AddressOfNewExeHeader = _size;
  }

  virtual void write(uint8_t *fileBuffer) {
    std::memcpy(fileBuffer, &_dosHeader, sizeof(_dosHeader));
  }

private:
  llvm::object::dos_header _dosHeader;
};

/// A PEHeaderChunk represents PE header including COFF header.
class PEHeaderChunk : public HeaderChunk {
public:
  PEHeaderChunk(const PECOFFTargetInfo &targetInfo) : HeaderChunk() {
    // Set the size of the chunk and initialize the header with null bytes.
    _size = sizeof(llvm::COFF::PEMagic) + sizeof(_coffHeader)
        + sizeof(_peHeader);
    std::memset(&_coffHeader, 0, sizeof(_coffHeader));
    std::memset(&_peHeader, 0, sizeof(_peHeader));

    _coffHeader.Machine = llvm::COFF::IMAGE_FILE_MACHINE_I386;
    _coffHeader.TimeDateStamp = time(NULL);

    // The size of PE header including optional data directory is always 224.
    _coffHeader.SizeOfOptionalHeader = 224;

    // Attributes of the executable. We set IMAGE_FILE_RELOCS_STRIPPED flag
    // because we do not support ".reloc" section. That means that the
    // executable will have to be loaded at the preferred address as specified
    // by ImageBase (which the Windows loader usually do), or fail to start
    // because of lack of relocation info.
    _coffHeader.Characteristics = llvm::COFF::IMAGE_FILE_32BIT_MACHINE |
                                  llvm::COFF::IMAGE_FILE_EXECUTABLE_IMAGE |
                                  llvm::COFF::IMAGE_FILE_RELOCS_STRIPPED;

    // 0x10b indicates a normal PE32 executable. For PE32+ it should be 0x20b.
    _peHeader.Magic = 0x10b;

    // The address of entry point relative to ImageBase. Windows executable
    // usually starts at address 0x401000.
    _peHeader.AddressOfEntryPoint = 0x1000;

    // The address of the executable when loaded into memory. The default for
    // DLLs is 0x10000000. The default for executables is 0x400000.
    _peHeader.ImageBase = IMAGE_BASE;

    // Sections should be page-aligned when loaded into memory, which is 4KB on
    // x86.
    _peHeader.SectionAlignment = PAGE_SIZE;

    // Sections in an executable file on disk should be sector-aligned (512 byte).
    _peHeader.FileAlignment = SECTOR_SIZE;

    // The required Windows version number. This is the internal version and
    // shouldn't be confused with product name. Windows 7 is version 6.1 and
    // Windows 8 is 6.2, for example.
    PECOFFTargetInfo::OSVersion minOSVersion = targetInfo.getMinOSVersion();
    _peHeader.MajorOperatingSystemVersion = minOSVersion.majorVersion;
    _peHeader.MinorOperatingSystemVersion = minOSVersion.minorVersion;
    _peHeader.MajorSubsystemVersion = minOSVersion.majorVersion;
    _peHeader.MinorSubsystemVersion = minOSVersion.minorVersion;

    // The combined size of the DOS, PE and section headers including garbage
    // between the end of the header and the beginning of the first section.
    // Must be multiple of FileAlignment.
    _peHeader.SizeOfHeaders = 512;
    _peHeader.Subsystem = targetInfo.getSubsystem();

    // Despite its name, DLL characteristics field has meaning both for
    // executables and DLLs. We are not very sure if the following bits must
    // be set, but regular binaries seem to have these bits, so we follow
    // them.
    uint16_t dllCharacteristics =
        llvm::COFF::IMAGE_DLL_CHARACTERISTICS_DYNAMIC_BASE |
        llvm::COFF::IMAGE_DLL_CHARACTERISTICS_NO_SEH |
        llvm::COFF::IMAGE_DLL_CHARACTERISTICS_TERMINAL_SERVER_AWARE;
    if (targetInfo.getNxCompat())
      dllCharacteristics |= llvm::COFF::IMAGE_DLL_CHARACTERISTICS_NX_COMPAT;
    _peHeader.DLLCharacteristics = dllCharacteristics;

    _peHeader.SizeOfStackReserve = targetInfo.getStackReserve();
    _peHeader.SizeOfStackCommit = targetInfo.getStackCommit();
    _peHeader.SizeOfHeapReserve = targetInfo.getHeapReserve();
    _peHeader.SizeOfHeapCommit = targetInfo.getHeapCommit();

    // The number of data directory entries. We always have 16 entries.
    _peHeader.NumberOfRvaAndSize = 16;
  }

  virtual void write(uint8_t *fileBuffer) {
    fileBuffer += fileOffset();
    std::memcpy(fileBuffer, llvm::COFF::PEMagic, sizeof(llvm::COFF::PEMagic));
    fileBuffer += sizeof(llvm::COFF::PEMagic);
    std::memcpy(fileBuffer, &_coffHeader, sizeof(_coffHeader));
    fileBuffer += sizeof(_coffHeader);
    std::memcpy(fileBuffer, &_peHeader, sizeof(_peHeader));
  }

  virtual void setSizeOfCode(uint64_t size) {
    _peHeader.SizeOfCode = size;
  }

  virtual void setNumberOfSections(uint32_t num) {
    _coffHeader.NumberOfSections = num;
  }

  virtual void setBaseOfCode(uint32_t rva) { _peHeader.BaseOfCode = rva; }

  virtual void setBaseOfData(uint32_t rva) { _peHeader.BaseOfData = rva; }

  virtual void setSizeOfImage(uint32_t size) { _peHeader.SizeOfImage = size; }

private:
  llvm::object::coff_file_header _coffHeader;
  llvm::object::pe32_header _peHeader;
};

/// A DataDirectoryChunk represents data directory entries that follows the PE
/// header in the output file. An entry consists of an 8 byte field that
/// indicates a relative virtual address (the starting address of the entry data
/// in memory) and 8 byte entry data size.
class DataDirectoryChunk : public HeaderChunk {
public:
  DataDirectoryChunk() : HeaderChunk() {
    // [FIXME] Currently all entries are filled with zero.
    _size = sizeof(_dirs);
    std::memset(&_dirs, 0, sizeof(_dirs));
  }

  virtual void write(uint8_t *fileBuffer) {
    fileBuffer += fileOffset();
    std::memcpy(fileBuffer, &_dirs, sizeof(_dirs));
  }

private:
  llvm::object::data_directory _dirs[16];
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

/// A SectionChunk represents a section containing atoms. It consists of a
/// section header that to be written to PECOFF header and atoms which to be
/// written to the raw data section.
class SectionChunk : public Chunk {
public:
  /// Returns the size of the section on disk. The returned value is multiple
  /// of disk sector, so the size may include the null padding at the end of
  /// section.
  virtual uint64_t size() const {
    return llvm::RoundUpToAlignment(_size, _align);
  }

  virtual void write(uint8_t *fileBuffer) {
    for (const auto *layout : _atomLayouts) {
      const DefinedAtom *atom = dyn_cast<const DefinedAtom>(layout->_atom);
      ArrayRef<uint8_t> rawContent = atom->rawContent();
      std::memcpy(fileBuffer + layout->_fileOffset, rawContent.data(),
                  rawContent.size());
    }
  }

  /// Add all atoms to the given map. This data will be used to do relocation.
  void
  buildAtomToVirtualAddr(std::map<const Atom *, uint64_t> &atomToVirtualAddr) {
    for (const auto *layout : _atomLayouts)
      atomToVirtualAddr[layout->_atom] = layout->_virtualAddr;
  }

  void applyRelocations(uint8_t *fileBuffer,
                        std::map<const Atom *, uint64_t> &atomToVirtualAddr) {
    for (const auto *layout : _atomLayouts) {
      const DefinedAtom *atom = dyn_cast<const DefinedAtom>(layout->_atom);
      for (const Reference *ref : *atom) {
        auto relocSite = reinterpret_cast<llvm::support::ulittle32_t *>(
            fileBuffer + layout->_fileOffset + ref->offsetInAtom());
        uint64_t targetAddr = atomToVirtualAddr[ref->target()];

        // Skip if this reference is not for relocation.
        if (ref->kind() < lld::Reference::kindTargetLow)
          continue;

        switch (ref->kind()) {
        case llvm::COFF::IMAGE_REL_I386_ABSOLUTE:
          // This relocation is no-op.
          break;
        case llvm::COFF::IMAGE_REL_I386_DIR32:
          // Set target's 32-bit VA.
          *relocSite = targetAddr + IMAGE_BASE;
          break;
        case llvm::COFF::IMAGE_REL_I386_DIR32NB:
          // Set target's 32-bit RVA.
          *relocSite = targetAddr;
          break;
        case llvm::COFF::IMAGE_REL_I386_REL32: {
          // Set 32-bit relative address of the target. This relocation is
          // usually used for relative branch or call instruction.
          uint32_t disp = atomToVirtualAddr[atom] + ref->offsetInAtom() + 4;
          *relocSite = targetAddr - disp;
          break;
        }
        default:
          llvm_unreachable("Unsupported relocation kind");
        }
      }
    }
  }

  // Set the file offset of the beginning of this section.
  virtual void setFileOffset(uint64_t fileOffset) {
    Chunk::setFileOffset(fileOffset);
    _sectionHeader.PointerToRawData = fileOffset;
    for (AtomLayout *layout : _atomLayouts)
      layout->_fileOffset += fileOffset;
  }

  virtual void setVirtualAddress(uint32_t rva) {
    _sectionHeader.VirtualAddress = rva;
    for (AtomLayout *layout : _atomLayouts)
      layout->_virtualAddr += rva;
  }

  virtual uint32_t getVirtualAddress() { return _sectionHeader.VirtualAddress; }

  const llvm::object::coff_section &getSectionHeader() {
    return _sectionHeader;
  }

  static bool classof(const Chunk *c) { return c->getKind() == kindSection; }

protected:
  SectionChunk(SectionHeaderTableChunk *table, StringRef sectionName,
               uint32_t characteristics)
      : Chunk(kindSection),
        _sectionHeader(createSectionHeader(sectionName, characteristics)) {
    // The section should be aligned to disk sector.
    _align = SECTOR_SIZE;

    // Add this section to the file header.
    table->addSection(this);
  }

  void buildContents(const File &linkedFile,
                     bool (*isEligible)(const DefinedAtom *)) {
    // Extract atoms from the linked file and append them to this section.
    for (const DefinedAtom *atom : linkedFile.defined()) {
      assert(atom->sectionChoice() == DefinedAtom::sectionBasedOnContent);
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

private:
  llvm::object::coff_section
  createSectionHeader(StringRef sectionName, uint32_t characteristics) const {
    llvm::object::coff_section header;

    // Section name equal to or shorter than 8 byte fits in the section
    // header. Longer names should be stored to string table, which is not
    // implemented yet.
    if (sizeof(header.Name) < sectionName.size())
      llvm_unreachable("Cannot handle section name longer than 8 byte");

    // Name field must be NUL-padded. If the name is exactly 8 byte long,
    // there's no terminating NUL.
    std::memset(header.Name, 0, sizeof(header.Name));
    std::strncpy(header.Name, sectionName.data(), sizeof(header.Name));

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

  void appendAtom(const DefinedAtom *atom) {
    auto *layout = new (_storage) AtomLayout(atom, _size, _size);
    _atomLayouts.push_back(layout);
    _size += atom->rawContent().size();
  }

  llvm::object::coff_section _sectionHeader;
  std::vector<AtomLayout *> _atomLayouts;
  mutable llvm::BumpPtrAllocator _storage;
};

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
    // Skip the empty section. Windows loader does not like a section
    // of size zero and rejects such executable.
    if (chunk->size() == 0)
      continue;

    const llvm::object::coff_section &header = chunk->getSectionHeader();
    std::memcpy(fileBuffer + offset, &header, sizeof(header));
    offset += sizeof(header);
  }
}

// \brief A TextSectionChunk represents a .text section.
class TextSectionChunk : public SectionChunk {
public:
  TextSectionChunk(const File &linkedFile, SectionHeaderTableChunk *table)
      : SectionChunk(table, ".text", characteristics) {
    buildContents(linkedFile, [](const DefinedAtom *atom) {
      return atom->contentType() == DefinedAtom::typeCode;
    });
  }

private:
  // When loaded into memory, text section should be readable and executable.
  static const uint32_t characteristics =
      llvm::COFF::IMAGE_SCN_CNT_CODE | llvm::COFF::IMAGE_SCN_MEM_EXECUTE |
      llvm::COFF::IMAGE_SCN_MEM_READ;
};

// \brief A RDataSectionChunk represents a .rdata section.
class RDataSectionChunk : public SectionChunk {
public:
  RDataSectionChunk(const File &linkedFile, SectionHeaderTableChunk *table)
      : SectionChunk(table, ".rdata", characteristics) {
    buildContents(linkedFile, [](const DefinedAtom *atom) {
      return (atom->contentType() == DefinedAtom::typeData &&
              atom->permissions() == DefinedAtom::permR__);
    });
  }

private:
  // When loaded into memory, rdata section should be readable.
  static const uint32_t characteristics =
      llvm::COFF::IMAGE_SCN_MEM_READ |
      llvm::COFF::IMAGE_SCN_CNT_INITIALIZED_DATA;
};

// \brief A DataSectionChunk represents a .data section.
class DataSectionChunk : public SectionChunk {
public:
  DataSectionChunk(const File &linkedFile, SectionHeaderTableChunk *table)
      : SectionChunk(table, ".data", characteristics) {
    buildContents(linkedFile, [](const DefinedAtom *atom) {
      return (atom->contentType() == DefinedAtom::typeData &&
              atom->permissions() == DefinedAtom::permRW_);
    });
  }

private:
  // When loaded into memory, data section should be readable and writable.
  static const uint32_t characteristics =
      llvm::COFF::IMAGE_SCN_MEM_READ |
      llvm::COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
      llvm::COFF::IMAGE_SCN_MEM_WRITE;
};

}  // end anonymous namespace

class ExecutableWriter : public Writer {
private:
  // Compute and set the offset of each chunk in the output file.
  void computeChunkSizeOnDisk() {
    uint64_t offset = 0;
    for (auto &chunk : _chunks) {
      // Round up to the nearest alignment boundary.
      offset = llvm::RoundUpToAlignment(offset, chunk->align());
      chunk->setFileOffset(offset);
      offset += chunk->size();
    }
  }

  // Compute the starting address of sections when loaded in memory. They are
  // different from positions on disk because sections need to be
  // sector-aligned on disk but page-aligned in memory.
  void computeChunkSizeInMemory(uint32_t &numSections, uint32_t &imageSize) {
    // The first page starting at ImageBase is usually left unmapped. IIUC
    // there's no technical reason to do so, but we'll follow that convention
    // so that we don't produce odd-looking binary. We should update the code
    // (or this comment) once we figure the reason out.
    uint32_t offset = PAGE_SIZE;
    uint32_t va = offset;
    for (auto &cp : _chunks) {
      if (SectionChunk *chunk = dyn_cast<SectionChunk>(&*cp)) {
        chunk->setVirtualAddress(va);

        // Skip the empty section.
        if (chunk->size() == 0)
          continue;
        numSections++;
        va = llvm::RoundUpToAlignment(va + chunk->size(), PAGE_SIZE);
      }
    }
    imageSize = va - offset;
  }

  /// Apply relocations to the output file buffer. This two pass. In the first
  /// pass, we visit all atoms to create a map from atom to its virtual
  /// address. In the second pass, we visit all relocation references to fix
  /// up addresses in the buffer.
  void applyRelocations(uint8_t *bufferStart) {
    std::map<const Atom *, uint64_t> atomToVirtualAddr;
    for (auto &cp : _chunks)
      if (SectionChunk *chunk = dyn_cast<SectionChunk>(&*cp))
        chunk->buildAtomToVirtualAddr(atomToVirtualAddr);
    for (auto &cp : _chunks)
      if (SectionChunk *chunk = dyn_cast<SectionChunk>(&*cp))
        chunk->applyRelocations(bufferStart, atomToVirtualAddr);
  }

  void addChunk(Chunk *chunk) {
    _chunks.push_back(std::unique_ptr<Chunk>(chunk));
  }

public:
  ExecutableWriter(const PECOFFTargetInfo &targetInfo)
      : _PECOFFTargetInfo(targetInfo) {}

  // Create all chunks that consist of the output file.
  void build(const File &linkedFile) {
    // Create file chunks and add them to the list.
    auto *dosStub = new DOSStubChunk();
    auto *peHeader = new PEHeaderChunk(_PECOFFTargetInfo);
    auto *dataDirectory = new DataDirectoryChunk();
    auto *sectionTable = new SectionHeaderTableChunk();
    auto *text = new TextSectionChunk(linkedFile, sectionTable);
    auto *rdata = new RDataSectionChunk(linkedFile, sectionTable);
    auto *data = new DataSectionChunk(linkedFile, sectionTable);

    addChunk(dosStub);
    addChunk(peHeader);
    addChunk(dataDirectory);
    addChunk(sectionTable);
    addChunk(text);
    addChunk(rdata);
    addChunk(data);

    // Compute and assign file offset to each chunk.
    uint32_t numSections = 0;
    uint32_t imageSize = 0;
    computeChunkSizeOnDisk();
    computeChunkSizeInMemory(numSections, imageSize);

    // Now that we know the size and file offset of sections. Set the file
    // header accordingly.
    peHeader->setSizeOfCode(text->size());
    peHeader->setBaseOfCode(text->getVirtualAddress());
    peHeader->setBaseOfData(rdata->getVirtualAddress());
    peHeader->setNumberOfSections(numSections);
    peHeader->setSizeOfImage(imageSize);
  }

  virtual error_code writeFile(const File &linkedFile, StringRef path) {
    this->build(linkedFile);

    uint64_t totalSize = _chunks.back()->fileOffset() + _chunks.back()->size();
    OwningPtr<llvm::FileOutputBuffer> buffer;
    error_code ec = llvm::FileOutputBuffer::create(
        path, totalSize, buffer, llvm::FileOutputBuffer::F_executable);
    if (ec)
      return ec;

    for (const auto &chunk : _chunks)
      chunk->write(buffer->getBufferStart());
    applyRelocations(buffer->getBufferStart());
    return buffer->commit();
  }

private:
  std::vector<std::unique_ptr<Chunk>> _chunks;
  const PECOFFTargetInfo &_PECOFFTargetInfo;
};

} // end namespace pecoff

std::unique_ptr<Writer> createWriterPECOFF(const PECOFFTargetInfo &info) {
  return std::unique_ptr<Writer>(new pecoff::ExecutableWriter(info));
}

} // end namespace lld
