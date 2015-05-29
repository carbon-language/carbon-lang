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
#include "lld/Core/Error.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <functional>
#include <map>
#include <utility>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::COFF;

static const int PageSize = 4096;
static const int FileAlignment = 512;
static const int SectionAlignment = 4096;
static const int DOSStubSize = 64;
static const int NumberfOfDataDirectory = 16;
static const int HeaderSize =
    DOSStubSize + sizeof(PEMagic) + sizeof(coff_file_header) +
    sizeof(pe32plus_header) + sizeof(data_directory) * NumberfOfDataDirectory;

namespace lld {
namespace coff {

OutputSection::OutputSection(StringRef N, uint32_t SI)
    : Name(N), SectionIndex(SI) {
  memset(&Header, 0, sizeof(Header));
  strncpy(Header.Name, Name.data(), std::min(Name.size(), size_t(8)));
}

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

void Writer::markLive() {
  Entry = cast<Defined>(Symtab->find(Config->EntryName));
  Entry->markLive();
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
    auto Sec =
        llvm::make_unique<OutputSection>(SectionName, OutputSections.size());
    for (Chunk *C : Chunks) {
      C->setOutputSection(Sec.get());
      Sec->addChunk(C);
      Sec->addPermissions(C->getPermissions());
    }
    OutputSections.push_back(std::move(Sec));
  }
}

std::map<StringRef, std::vector<DefinedImportData *>> Writer::binImports() {
  // Group DLL-imported symbols by DLL name because that's how symbols
  // are layed out in the import descriptor table.
  std::map<StringRef, std::vector<DefinedImportData *>> Res;
  OutputSection *Text = createSection(".text");
  for (std::unique_ptr<ImportFile> &P : Symtab->ImportFiles) {
    for (SymbolBody *B : P->getSymbols()) {
      if (auto *Import = dyn_cast<DefinedImportData>(B)) {
        Res[Import->getDLLName()].push_back(Import);
        continue;
      }
      // Linker-created function thunks for DLL symbols are added to
      // .text section.
      Text->addChunk(cast<DefinedImportThunk>(B)->getChunk());
    }
  }

  // Sort symbols by name for each group.
  auto Comp = [](DefinedImportData *A, DefinedImportData *B) {
    return A->getName() < B->getName();
  };
  for (auto &P : Res) {
    std::vector<DefinedImportData *> &V = P.second;
    std::sort(V.begin(), V.end(), Comp);
  }
  return Res;
}

// Create .idata section contents.
void Writer::createImportTables() {
  if (Symtab->ImportFiles.empty())
    return;

  std::vector<ImportTable> Tabs;
  for (auto &P : binImports()) {
    StringRef DLLName = P.first;
    std::vector<DefinedImportData *> &Imports = P.second;
    Tabs.emplace_back(DLLName, Imports);
  }
  OutputSection *Idata = createSection(".idata");
  size_t NumChunks = Idata->getChunks().size();

  // Add the directory tables.
  for (ImportTable &T : Tabs)
    Idata->addChunk(T.DirTab);
  Idata->addChunk(new NullChunk(sizeof(ImportDirectoryTableEntry)));
  ImportDirectoryTableSize = (Tabs.size() + 1) * sizeof(ImportDirectoryTableEntry);

  // Add the import lookup tables.
  for (ImportTable &T : Tabs) {
    for (LookupChunk *C : T.LookupTables)
      Idata->addChunk(C);
    Idata->addChunk(new NullChunk(sizeof(uint64_t)));
  }

  // Add the import address tables. Their contents are the same as the
  // lookup tables.
  for (ImportTable &T : Tabs) {
    for (LookupChunk *C : T.AddressTables)
      Idata->addChunk(C);
    Idata->addChunk(new NullChunk(sizeof(uint64_t)));
    ImportAddressTableSize += (T.AddressTables.size() + 1) * sizeof(uint64_t);
  }
  ImportAddressTable = Tabs[0].AddressTables[0];

  // Add the hint name table.
  for (ImportTable &T : Tabs)
    for (HintNameChunk *C : T.HintNameTables)
      Idata->addChunk(C);

  // Add DLL names.
  for (ImportTable &T : Tabs)
    Idata->addChunk(T.DLLName);

  // Claim ownership of all chunks in the .idata section.
  for (size_t I = NumChunks, E = Idata->getChunks().size(); I < E; ++I)
    Chunks.push_back(std::unique_ptr<Chunk>(Idata->getChunks()[I]));
}

// The Windows loader doesn't seem to like empty sections,
// so we remove them if any.
void Writer::removeEmptySections() {
  auto IsEmpty = [](const std::unique_ptr<OutputSection> &S) {
    return S->getVirtualSize() == 0;
  };
  OutputSections.erase(
      std::remove_if(OutputSections.begin(), OutputSections.end(), IsEmpty),
      OutputSections.end());
}

// Visits all sections to assign incremental, non-overlapping RVAs and
// file offsets.
void Writer::assignAddresses() {
  SizeOfHeaders = RoundUpToAlignment(
      HeaderSize + sizeof(coff_section) * OutputSections.size(), PageSize);
  uint64_t RVA = 0x1000; // The first page is kept unmapped.
  uint64_t FileOff = SizeOfHeaders;
  for (std::unique_ptr<OutputSection> &Sec : OutputSections) {
    Sec->setRVA(RVA);
    Sec->setFileOffset(FileOff);
    RVA += RoundUpToAlignment(Sec->getVirtualSize(), PageSize);
    FileOff += RoundUpToAlignment(Sec->getRawSize(), FileAlignment);
  }
  SizeOfImage = SizeOfHeaders + RoundUpToAlignment(RVA - 0x1000, PageSize);
  FileSize = SizeOfHeaders +
             RoundUpToAlignment(FileOff - SizeOfHeaders, FileAlignment);
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

  // Write COFF header
  auto *COFF = reinterpret_cast<coff_file_header *>(Buf);
  Buf += sizeof(*COFF);
  COFF->Machine = Config->MachineType;
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
  PE->MajorOperatingSystemVersion = 6;
  PE->MajorSubsystemVersion = 6;
  PE->Subsystem = IMAGE_SUBSYSTEM_WINDOWS_CUI;
  PE->SizeOfImage = SizeOfImage;
  PE->SizeOfHeaders = SizeOfHeaders;
  PE->AddressOfEntryPoint = Entry->getRVA();
  PE->SizeOfStackReserve = 1024 * 1024;
  PE->SizeOfStackCommit = 4096;
  PE->SizeOfHeapReserve = 1024 * 1024;
  PE->SizeOfHeapCommit = 4096;
  PE->NumberOfRvaAndSize = NumberfOfDataDirectory;
  if (OutputSection *Text = findSection(".text")) {
    PE->BaseOfCode = Text->getRVA();
    PE->SizeOfCode = Text->getRawSize();
  }
  PE->SizeOfInitializedData = getSizeOfInitializedData();

  // Write data directory
  auto *DataDirectory = reinterpret_cast<data_directory *>(Buf);
  Buf += sizeof(*DataDirectory) * NumberfOfDataDirectory;
  if (OutputSection *Idata = findSection(".idata")) {
    using namespace llvm::COFF;
    DataDirectory[IMPORT_TABLE].RelativeVirtualAddress = Idata->getRVA();
    DataDirectory[IMPORT_TABLE].Size = ImportDirectoryTableSize;
    DataDirectory[IAT].RelativeVirtualAddress = ImportAddressTable->getRVA();
    DataDirectory[IAT].Size = ImportAddressTableSize;
  }

  // Write section table
  coff_section *SectionTable = reinterpret_cast<coff_section *>(Buf);
  int Idx = 0;
  for (std::unique_ptr<OutputSection> &Sec : OutputSections)
    SectionTable[Idx++] = Sec->getHeader();
}

std::error_code Writer::openFile(StringRef Path) {
  if (auto EC = FileOutputBuffer::create(Path, FileSize, Buffer,
                                         FileOutputBuffer::F_executable))
    return make_dynamic_error_code(Twine("Failed to open ") + Path + ": " +
                                   EC.message());
  return std::error_code();
}

// Write section contents to a mmap'ed file.
void Writer::writeSections() {
  uint8_t *Buf = Buffer->getBufferStart();
  for (std::unique_ptr<OutputSection> &Sec : OutputSections) {
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
  for (std::unique_ptr<OutputSection> &Sec : OutputSections)
    if (Sec->getName() == Name)
      return Sec.get();
  return nullptr;
}

uint32_t Writer::getSizeOfInitializedData() {
  uint32_t Res = 0;
  for (std::unique_ptr<OutputSection> &S : OutputSections)
    if (S->getPermissions() & IMAGE_SCN_CNT_INITIALIZED_DATA)
      Res += S->getRawSize();
  return Res;
}

// Returns an existing section or create a new one if not found.
OutputSection *Writer::createSection(StringRef Name) {
  if (auto *Sec = findSection(Name))
    return Sec;
  const auto R = IMAGE_SCN_MEM_READ;
  const auto W = IMAGE_SCN_MEM_WRITE;
  const auto E = IMAGE_SCN_MEM_EXECUTE;
  uint32_t Perm = StringSwitch<uint32_t>(Name)
                      .Case(".bss", IMAGE_SCN_CNT_UNINITIALIZED_DATA | R | W)
                      .Case(".data", IMAGE_SCN_CNT_INITIALIZED_DATA | R | W)
                      .Case(".idata", IMAGE_SCN_CNT_INITIALIZED_DATA | R)
                      .Case(".rdata", IMAGE_SCN_CNT_INITIALIZED_DATA | R)
                      .Case(".text", IMAGE_SCN_CNT_CODE | R | E)
                      .Default(0);
  if (!Perm)
    llvm_unreachable("unknown section name");
  auto Sec = new OutputSection(Name, OutputSections.size());
  Sec->addPermissions(Perm);
  OutputSections.push_back(std::unique_ptr<OutputSection>(Sec));
  return Sec;
}

void Writer::applyRelocations() {
  uint8_t *Buf = Buffer->getBufferStart();
  for (std::unique_ptr<OutputSection> &Sec : OutputSections)
    for (Chunk *C : Sec->getChunks())
      C->applyRelocations(Buf);
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
  applyRelocations();
  return Buffer->commit();
}

} // namespace coff
} // namespace lld
