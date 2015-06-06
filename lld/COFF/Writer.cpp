//===- Writer.cpp ---------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Config.h"
#include "Writer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdio>
#include <functional>
#include <map>
#include <utility>

using namespace llvm;
using namespace llvm::COFF;
using namespace llvm::object;
using namespace llvm::support;
using namespace llvm::support::endian;

static const int PageSize = 4096;
static const int FileAlignment = 512;
static const int SectionAlignment = 4096;
static const int DOSStubSize = 64;
static const int NumberfOfDataDirectory = 16;

namespace lld {
namespace coff {

void OutputSection::setRVA(uint64_t RVA) {
  Header.VirtualAddress = RVA;
  for (Chunk *C : Chunks)
    C->setRVA(C->getRVA() + RVA);
}

void OutputSection::setFileOffset(uint64_t Off) {
  // If a section has no actual data (i.e. BSS section), we want to
  // set 0 to its PointerToRawData. Otherwise the output is rejected
  // by the loader.
  if (Header.SizeOfRawData == 0)
    return;
  Header.PointerToRawData = Off;
  for (Chunk *C : Chunks)
    C->setFileOff(C->getFileOff() + Off);
}

void OutputSection::addChunk(Chunk *C) {
  Chunks.push_back(C);
  uint64_t Off = Header.VirtualSize;
  Off = RoundUpToAlignment(Off, C->getAlign());
  C->setRVA(Off);
  C->setFileOff(Off);
  Off += C->getSize();
  Header.VirtualSize = Off;
  if (C->hasData())
    Header.SizeOfRawData = RoundUpToAlignment(Off, FileAlignment);
}

void OutputSection::addPermissions(uint32_t C) {
  Header.Characteristics = Header.Characteristics | (C & PermMask);
}

// Write the section header to a given buffer.
void OutputSection::writeHeader(uint8_t *Buf) {
  auto *Hdr = reinterpret_cast<coff_section *>(Buf);
  *Hdr = Header;
  if (StringTableOff) {
    // If name is too long, write offset into the string table as a name.
    sprintf(Hdr->Name, "/%d", StringTableOff);
  } else {
    assert(Name.size() <= COFF::NameSize);
    strncpy(Hdr->Name, Name.data(), Name.size());
  }
}

void Writer::markLive() {
  for (StringRef Name : Config->GCRoots)
    cast<Defined>(Symtab->find(Name))->markLive();
  for (Chunk *C : Symtab->getChunks())
    if (C->isRoot())
      C->markLive();
}

void Writer::createSections() {
  std::map<StringRef, std::vector<Chunk *>> Map;
  for (Chunk *C : Symtab->getChunks()) {
    if (!C->isLive()) {
      if (Config->Verbose)
        C->printDiscardedMessage();
      continue;
    }
    // '$' and all following characters in input section names are
    // discarded when determining output section. So, .text$foo
    // contributes to .text, for example. See PE/COFF spec 3.2.
    Map[C->getSectionName().split('$').first].push_back(C);
  }

  // Input sections are ordered by their names including '$' parts,
  // which gives you some control over the output layout.
  auto Comp = [](Chunk *A, Chunk *B) {
    return A->getSectionName() < B->getSectionName();
  };
  for (auto &P : Map) {
    StringRef SectionName = P.first;
    std::vector<Chunk *> &Chunks = P.second;
    std::stable_sort(Chunks.begin(), Chunks.end(), Comp);
    size_t SectIdx = OutputSections.size();
    auto Sec = new (CAlloc.Allocate()) OutputSection(SectionName, SectIdx);
    for (Chunk *C : Chunks) {
      C->setOutputSection(Sec);
      Sec->addChunk(C);
      Sec->addPermissions(C->getPermissions());
    }
    OutputSections.push_back(Sec);
  }
}

// Create .idata section for the DLL-imported symbol table.
// The format of this section is inherently Windows-specific.
// IdataContents class abstracted away the details for us,
// so we just let it create chunks and add them to the section.
void Writer::createImportTables() {
  if (Symtab->ImportFiles.empty())
    return;
  OutputSection *Text = createSection(".text");
  Idata.reset(new IdataContents());
  for (std::unique_ptr<ImportFile> &File : Symtab->ImportFiles) {
    for (SymbolBody *Body : File->getSymbols()) {
      if (auto *Import = dyn_cast<DefinedImportData>(Body)) {
        Idata->add(Import);
        continue;
      }
      // Linker-created function thunks for DLL symbols are added to
      // .text section.
      Text->addChunk(cast<DefinedImportThunk>(Body)->getChunk());
    }
  }
  OutputSection *Sec = createSection(".idata");
  for (Chunk *C : Idata->getChunks())
    Sec->addChunk(C);
}

// The Windows loader doesn't seem to like empty sections,
// so we remove them if any.
void Writer::removeEmptySections() {
  auto IsEmpty = [](OutputSection *S) { return S->getVirtualSize() == 0; };
  OutputSections.erase(
      std::remove_if(OutputSections.begin(), OutputSections.end(), IsEmpty),
      OutputSections.end());
}

// Visits all sections to assign incremental, non-overlapping RVAs and
// file offsets.
void Writer::assignAddresses() {
  SizeOfHeaders = RoundUpToAlignment(
      DOSStubSize + sizeof(PEMagic) + sizeof(coff_file_header) +
      sizeof(pe32plus_header) +
      sizeof(data_directory) * NumberfOfDataDirectory +
      sizeof(coff_section) * OutputSections.size(), PageSize);
  uint64_t RVA = 0x1000; // The first page is kept unmapped.
  uint64_t FileOff = SizeOfHeaders;
  for (OutputSection *Sec : OutputSections) {
    Sec->setRVA(RVA);
    Sec->setFileOffset(FileOff);
    RVA += RoundUpToAlignment(Sec->getVirtualSize(), PageSize);
    FileOff += RoundUpToAlignment(Sec->getRawSize(), FileAlignment);
  }
  SizeOfImage = SizeOfHeaders + RoundUpToAlignment(RVA - 0x1000, PageSize);
  FileSize = SizeOfHeaders +
             RoundUpToAlignment(FileOff - SizeOfHeaders, FileAlignment);
}

static MachineTypes
inferMachineType(std::vector<std::unique_ptr<ObjectFile>> &ObjectFiles) {
  for (std::unique_ptr<ObjectFile> &File : ObjectFiles) {
    // Try to infer machine type from the magic byte of the object file.
    auto MT = static_cast<MachineTypes>(File->getCOFFObj()->getMachine());
    if (MT != IMAGE_FILE_MACHINE_UNKNOWN)
      return MT;
  }
  return IMAGE_FILE_MACHINE_UNKNOWN;
}

void Writer::writeHeader() {
  // Write DOS stub
  uint8_t *Buf = Buffer->getBufferStart();
  auto *DOS = reinterpret_cast<dos_header *>(Buf);
  Buf += DOSStubSize;
  DOS->Magic[0] = 'M';
  DOS->Magic[1] = 'Z';
  DOS->AddressOfRelocationTable = sizeof(dos_header);
  DOS->AddressOfNewExeHeader = DOSStubSize;

  // Write PE magic
  memcpy(Buf, PEMagic, sizeof(PEMagic));
  Buf += sizeof(PEMagic);

  // Determine machine type, infer if needed. TODO: diagnose conflicts.
  MachineTypes MachineType = Config->MachineType;
  if (MachineType == IMAGE_FILE_MACHINE_UNKNOWN)
    MachineType = inferMachineType(Symtab->ObjectFiles);

  // Write COFF header
  auto *COFF = reinterpret_cast<coff_file_header *>(Buf);
  Buf += sizeof(*COFF);
  COFF->Machine = MachineType;
  COFF->NumberOfSections = OutputSections.size();
  COFF->Characteristics =
      (IMAGE_FILE_EXECUTABLE_IMAGE | IMAGE_FILE_RELOCS_STRIPPED |
       IMAGE_FILE_LARGE_ADDRESS_AWARE);
  COFF->SizeOfOptionalHeader =
      sizeof(pe32plus_header) + sizeof(data_directory) * NumberfOfDataDirectory;

  // Write PE header
  auto *PE = reinterpret_cast<pe32plus_header *>(Buf);
  Buf += sizeof(*PE);
  PE->Magic = PE32Header::PE32_PLUS;
  PE->ImageBase = Config->ImageBase;
  PE->SectionAlignment = SectionAlignment;
  PE->FileAlignment = FileAlignment;
  PE->MajorImageVersion = Config->MajorImageVersion;
  PE->MinorImageVersion = Config->MinorImageVersion;
  PE->MajorOperatingSystemVersion = Config->MajorOSVersion;
  PE->MinorOperatingSystemVersion = Config->MinorOSVersion;
  PE->MajorSubsystemVersion = Config->MajorOSVersion;
  PE->MinorSubsystemVersion = Config->MinorOSVersion;
  PE->Subsystem = Config->Subsystem;
  PE->SizeOfImage = SizeOfImage;
  PE->SizeOfHeaders = SizeOfHeaders;
  Defined *Entry = cast<Defined>(Symtab->find(Config->EntryName));
  PE->AddressOfEntryPoint = Entry->getRVA();
  PE->SizeOfStackReserve = Config->StackReserve;
  PE->SizeOfStackCommit = Config->StackCommit;
  PE->SizeOfHeapReserve = Config->HeapReserve;
  PE->SizeOfHeapCommit = Config->HeapCommit;
  PE->NumberOfRvaAndSize = NumberfOfDataDirectory;
  if (OutputSection *Text = findSection(".text")) {
    PE->BaseOfCode = Text->getRVA();
    PE->SizeOfCode = Text->getRawSize();
  }
  PE->SizeOfInitializedData = getSizeOfInitializedData();

  // Write data directory
  auto *DataDirectory = reinterpret_cast<data_directory *>(Buf);
  Buf += sizeof(*DataDirectory) * NumberfOfDataDirectory;
  if (Idata) {
    DataDirectory[IMPORT_TABLE].RelativeVirtualAddress = Idata->getDirRVA();
    DataDirectory[IMPORT_TABLE].Size = Idata->getDirSize();
    DataDirectory[IAT].RelativeVirtualAddress = Idata->getIATRVA();
    DataDirectory[IAT].Size = Idata->getIATSize();
  }

  // Section table
  // Name field in the section table is 8 byte long. Longer names need
  // to be written to the string table. First, construct string table.
  std::vector<char> Strtab;
  for (OutputSection *Sec : OutputSections) {
    StringRef Name = Sec->getName();
    if (Name.size() <= COFF::NameSize)
      continue;
    Sec->setStringTableOff(Strtab.size() + 4); // +4 for the size field
    Strtab.insert(Strtab.end(), Name.begin(), Name.end());
    Strtab.push_back('\0');
  }

  // Write section table
  for (OutputSection *Sec : OutputSections) {
    Sec->writeHeader(Buf);
    Buf += sizeof(coff_section);
  }

  // Write string table if we need to. The string table immediately
  // follows the symbol table, so we create a dummy symbol table
  // first. The symbol table contains one dummy symbol.
  if (Strtab.empty())
    return;
  COFF->PointerToSymbolTable = Buf - Buffer->getBufferStart();
  COFF->NumberOfSymbols = 1;
  auto *SymbolTable = reinterpret_cast<coff_symbol16 *>(Buf);
  Buf += sizeof(*SymbolTable);
  // (Set 4 to make the dummy symbol point to the first string table
  // entry, so that tools to print out symbols don't read NUL bytes.)
  SymbolTable->Name.Offset.Offset = 4;
  // Then create the symbol table. The first 4 bytes is length
  // including itself.
  write32le(Buf, Strtab.size() + 4);
  memcpy(Buf + 4, Strtab.data(), Strtab.size());
}

std::error_code Writer::openFile(StringRef Path) {
  if (auto EC = FileOutputBuffer::create(Path, FileSize, Buffer,
                                         FileOutputBuffer::F_executable)) {
    llvm::errs() << "failed to open " << Path << ": " << EC.message() << "\n";
    return EC;
  }
  return std::error_code();
}

// Write section contents to a mmap'ed file.
void Writer::writeSections() {
  uint8_t *Buf = Buffer->getBufferStart();
  for (OutputSection *Sec : OutputSections) {
    // Fill gaps between functions in .text with INT3 instructions
    // instead of leaving as NUL bytes (which can be interpreted as
    // ADD instructions).
    if (Sec->getPermissions() & IMAGE_SCN_CNT_CODE)
      memset(Buf + Sec->getFileOff(), 0xCC, Sec->getRawSize());
    for (Chunk *C : Sec->getChunks())
      C->writeTo(Buf);
  }
}

OutputSection *Writer::findSection(StringRef Name) {
  for (OutputSection *Sec : OutputSections)
    if (Sec->getName() == Name)
      return Sec;
  return nullptr;
}

uint32_t Writer::getSizeOfInitializedData() {
  uint32_t Res = 0;
  for (OutputSection *S : OutputSections)
    if (S->getPermissions() & IMAGE_SCN_CNT_INITIALIZED_DATA)
      Res += S->getRawSize();
  return Res;
}

// Returns an existing section or create a new one if not found.
OutputSection *Writer::createSection(StringRef Name) {
  if (auto *Sec = findSection(Name))
    return Sec;
  const auto DATA = IMAGE_SCN_CNT_INITIALIZED_DATA;
  const auto BSS = IMAGE_SCN_CNT_UNINITIALIZED_DATA;
  const auto CODE = IMAGE_SCN_CNT_CODE;
  const auto R = IMAGE_SCN_MEM_READ;
  const auto W = IMAGE_SCN_MEM_WRITE;
  const auto E = IMAGE_SCN_MEM_EXECUTE;
  uint32_t Perms = StringSwitch<uint32_t>(Name)
                       .Case(".bss", BSS | R | W)
                       .Case(".data", DATA | R | W)
                       .Case(".didat", DATA | R)
                       .Case(".idata", DATA | R)
                       .Case(".rdata", DATA | R)
                       .Case(".text", CODE | R | E)
                       .Default(0);
  if (!Perms)
    llvm_unreachable("unknown section name");
  size_t SectIdx = OutputSections.size();
  auto Sec = new (CAlloc.Allocate()) OutputSection(Name, SectIdx);
  Sec->addPermissions(Perms);
  OutputSections.push_back(Sec);
  return Sec;
}

std::error_code Writer::write(StringRef OutputPath) {
  markLive();
  createSections();
  createImportTables();
  assignAddresses();
  removeEmptySections();
  if (auto EC = openFile(OutputPath))
    return EC;
  writeHeader();
  writeSections();
  return Buffer->commit();
}

} // namespace coff
} // namespace lld
