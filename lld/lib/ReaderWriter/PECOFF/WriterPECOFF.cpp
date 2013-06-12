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

#include <time.h>
#include <vector>

#include "lld/Core/DefinedAtom.h"
#include "lld/Core/File.h"
#include "lld/Core/InputFiles.h"
#include "lld/ReaderWriter/PECOFFTargetInfo.h"
#include "lld/ReaderWriter/Writer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileOutputBuffer.h"

namespace lld {
namespace pecoff {

namespace {

// Page size of x86 processor. Some data needs to be aligned at page boundary
// when loaded into memory.
const int PAGE_SIZE = 4096;

// Disk sector size. Some data needs to be aligned at disk sector boundary in
// file.
const int SECTOR_SIZE = 512;

/// A Chunk is an abstrace contiguous range in an output file.
class Chunk {
public:
  Chunk() : _size(0), _align(1) {}
  virtual ~Chunk() {};
  virtual void write(uint8_t *fileBuffer) = 0;

  virtual uint64_t fileOffset() const { return _fileOffset; }
  virtual uint64_t size() const { return _size; }
  virtual uint64_t align() const { return _align; }

  virtual void setFileOffset(uint64_t fileOffset) {
    _fileOffset = fileOffset;
  }

protected:
  uint64_t _size;
  uint64_t _fileOffset;
  uint64_t _align;
};

/// A DOSStubChunk represents the DOS compatible header at the beginning
/// of PE/COFF files.
class DOSStubChunk : public Chunk {
public:
  DOSStubChunk() : Chunk() {
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

/// A PEHeaderChunk represents PE header.
class PEHeaderChunk : public Chunk {
public:
  PEHeaderChunk(const PECOFFTargetInfo &targetInfo) : Chunk() {
    // Set the size of the chunk and initialize the header with null bytes.
    _size = sizeof(llvm::COFF::PEMagic) + sizeof(_coffHeader)
        + sizeof(_peHeader);
    std::memset(&_coffHeader, 0, sizeof(_coffHeader));
    std::memset(&_peHeader, 0, sizeof(_peHeader));


    _coffHeader.Machine = llvm::COFF::IMAGE_FILE_MACHINE_I386;
    _coffHeader.NumberOfSections = 1;  // [FIXME]
    _coffHeader.TimeDateStamp = time(NULL);

    // The size of PE header including optional data directory is always 224.
    _coffHeader.SizeOfOptionalHeader = 224;
    _coffHeader.Characteristics = llvm::COFF::IMAGE_FILE_32BIT_MACHINE
        | llvm::COFF::IMAGE_FILE_EXECUTABLE_IMAGE;

    // 0x10b indicates a normal executable. For PE32+ it should be 0x20b.
    _peHeader.Magic = 0x10b;

    // The address of entry point relative to ImageBase. Windows executable
    // usually starts at address 0x401000.
    _peHeader.AddressOfEntryPoint = 0x1000;
    _peHeader.BaseOfCode = 0x1000;

    // [FIXME] The address of data section relative to ImageBase.
    _peHeader.BaseOfData = 0x2000;

    // The address of the executable when loaded into memory. The default for
    // DLLs is 0x10000000. The default for executables is 0x400000.
    _peHeader.ImageBase = 0x400000;

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

    // [FIXME] The size of the image when loaded into memory
    _peHeader.SizeOfImage = 0x2000;

    // The combined size of the DOS, PE and section headers including garbage
    // between the end of the header and the beginning of the first section.
    // Must be multiple of FileAlignment.
    _peHeader.SizeOfHeaders = 512;
    _peHeader.Subsystem = targetInfo.getSubsystem();
    _peHeader.DLLCharacteristics =
        llvm::COFF::IMAGE_DLL_CHARACTERISTICS_DYNAMIC_BASE
        | llvm::COFF::IMAGE_DLL_CHARACTERISTICS_NX_COMPAT
        | llvm::COFF::IMAGE_DLL_CHARACTERISTICS_TERMINAL_SERVER_AWARE;

    _peHeader.SizeOfStackReserve = targetInfo.getStackReserve();
    _peHeader.SizeOfStackCommit = targetInfo.getStackCommit();
    _peHeader.SizeOfHeapReserve = targetInfo.getHeapReserve();
    _peHeader.SizeOfHeapCommit = targetInfo.getHeapCommit();

    // The number of data directory entries. We always have 16 entries.
    _peHeader.NumberOfRvaAndSize = 16;
  }

  virtual void write(uint8_t *fileBuffer) {
    std::memcpy(fileBuffer, llvm::COFF::PEMagic, sizeof(llvm::COFF::PEMagic));
    fileBuffer += sizeof(llvm::COFF::PEMagic);
    std::memcpy(fileBuffer, &_coffHeader, sizeof(_coffHeader));
    fileBuffer += sizeof(_coffHeader);
    std::memcpy(fileBuffer, &_peHeader, sizeof(_peHeader));
  }

  virtual void setSizeOfCode(uint64_t size) {
    _peHeader.SizeOfCode = size;
  }

private:
  llvm::object::coff_file_header _coffHeader;
  llvm::object::pe32_header _peHeader;
};

/// A DataDirectoryChunk represents data directory entries that follows the PE
/// header in the output file. An entry consists of an 8 byte field that
/// indicates a relative virtual address (the starting address of the entry data
/// in memory) and 8 byte entry data size.
class DataDirectoryChunk : public Chunk {
public:
  DataDirectoryChunk() : Chunk() {
    // [FIXME] Currently all entries are filled with zero.
    _size = sizeof(_dirs);
    std::memset(&_dirs, 0, sizeof(_dirs));
  }

  virtual void write(uint8_t *fileBuffer) {
    std::memcpy(fileBuffer, &_dirs, sizeof(_dirs));
  }

private:
  llvm::object::data_directory _dirs[16];
};

/// A SectionChunk represents a section in the output file. It consists of a
/// section header and atoms which to be output as the content of the section.
class SectionChunk : public Chunk {
public:
  SectionChunk(llvm::object::coff_section sectionHeader)
      : _sectionHeader(sectionHeader) {}

  void appendAtom(const DefinedAtom *atom) {
    _atoms.push_back(atom);
    _size += atom->rawContent().size();
  }

  virtual void write(uint8_t *fileBuffer) {
    uint64_t offset = 0;
    for (const auto &atom : _atoms) {
      ArrayRef<uint8_t> rawContent = atom->rawContent();
      std::memcpy(fileBuffer + offset, rawContent.data(), rawContent.size());
      offset += rawContent.size();
    }
  }

  const llvm::object::coff_section &getSectionHeader() {
    return _sectionHeader;
  }

protected:
  llvm::object::coff_section _sectionHeader;

private:
  std::vector<const DefinedAtom *> _atoms;
};

/// A SectionHeaderTableChunk is a list of section headers. The number of
/// section headers is in the PE header. A section header has metadata about the
/// section and a file offset to its content. Each section header is 40 byte and
/// contiguous in the output file.
class SectionHeaderTableChunk : public Chunk {
public:
  SectionHeaderTableChunk() : Chunk() {}

  void addSection(SectionChunk *chunk) {
    _sections.push_back(chunk);
  }

  virtual uint64_t size() const {
    return _sections.size() * sizeof(llvm::object::coff_section);
  }

  virtual void write(uint8_t *fileBuffer) {
    uint64_t offset = 0;
    for (const auto &chunk : _sections) {
      const llvm::object::coff_section &header = chunk->getSectionHeader();
      std::memcpy(fileBuffer + offset, &header, sizeof(header));
      offset += sizeof(header);
    }
  }

private:
  std::vector<SectionChunk*> _sections;
};

// \brief A TextSectionChunk represents a .text section.
class TextSectionChunk : public SectionChunk {
private:
  llvm::object::coff_section createSectionHeader() {
    llvm::object::coff_section header;
    std::memcpy(&header.Name, ".text\0\0\0\0", 8);
    header.VirtualSize = 0;
    header.VirtualAddress = 0x1000;
    header.SizeOfRawData = 0;
    header.PointerToRawData = 0;
    header.PointerToRelocations = 0;
    header.PointerToLinenumbers = 0;
    header.NumberOfRelocations = 0;
    header.NumberOfLinenumbers = 0;
    header.Characteristics = llvm::COFF::IMAGE_SCN_CNT_CODE
        | llvm::COFF::IMAGE_SCN_MEM_EXECUTE
        | llvm::COFF::IMAGE_SCN_MEM_READ;
    return header;
  }

public:
  TextSectionChunk(const File &linkedFile)
      : SectionChunk(createSectionHeader()) {
    // The text section should be aligned to disk sector.
    _align = SECTOR_SIZE;

    // Extract executable atoms from the linked file and append them to this
    // section.
    for (const DefinedAtom* atom : linkedFile.defined()) {
      assert(atom->sectionChoice() == DefinedAtom::sectionBasedOnContent);
      DefinedAtom::ContentType type = atom->contentType();
      if (type != DefinedAtom::typeCode)
        continue;
      appendAtom(atom);
    }

    // Now that we have a list of atoms that to be written in this section, and
    // we know the size of the section.
    _sectionHeader.VirtualSize = _size;
    _sectionHeader.SizeOfRawData = _size;
  }

  virtual uint64_t size() const {
    // Round up to the nearest alignment border, so that the text segment ends
    // at a border.
    return (_size + _align - 1) & -_align;
  }

  // Set the file offset of the beginning of this section.
  virtual void setFileOffset(uint64_t fileOffset) {
    SectionChunk::setFileOffset(fileOffset);
    _sectionHeader.PointerToRawData = fileOffset;
  }
};

};  // end anonymous namespace

class ExecutableWriter : public Writer {
private:
  // Compute and set the offset of each chunk in the output file.
  void computeChunkSize() {
    uint64_t offset = 0;
    for (auto &chunk : _chunks) {
      // Round up to the nearest alignment boundary.
      offset = (offset + chunk->align() - 1) & -chunk->align();
      chunk->setFileOffset(offset);
      offset += chunk->size();
    }
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
    Chunk *dosStub(new DOSStubChunk());
    PEHeaderChunk *peHeader(new PEHeaderChunk(_PECOFFTargetInfo));
    Chunk *dataDirectoryHeader(new DataDirectoryChunk());
    SectionHeaderTableChunk *sectionTable(new SectionHeaderTableChunk());
    addChunk(dosStub);
    addChunk(peHeader);
    addChunk(dataDirectoryHeader);
    addChunk(sectionTable);

    // Create text section.
    // [FIXME] Handle data and bss sections.
    SectionChunk *text = new TextSectionChunk(linkedFile);
    sectionTable->addSection(text);
    addChunk(text);

    // Compute and assign file offset to each chunk.
    computeChunkSize();

    // Now that we know the size and file offset of sections. Set the file
    // header accordingly.
    peHeader->setSizeOfCode(text->size());
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
      chunk->write(buffer->getBufferStart() + chunk->fileOffset());
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
