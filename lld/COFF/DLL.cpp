//===- DLL.cpp ------------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines various types of chunks for the DLL import
// descriptor table. They are inherently Windows-specific.
// You need to read Microsoft PE/COFF spec to understand details
// about the data structures.
//
// If you are not particularly interested in linking against Windows
// DLL, you can skip this file, and you should still be able to
// understand the rest of the linker.
//
//===----------------------------------------------------------------------===//

#include "Chunks.h"
#include "DLL.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::COFF;
using llvm::RoundUpToAlignment;

static const size_t LookupChunkSize = sizeof(uint64_t);

namespace lld {
namespace coff {

// Import table

// A chunk for the import descriptor table.
class HintNameChunk : public Chunk {
public:
  HintNameChunk(StringRef N, uint16_t H) : Name(N), Hint(H) {}

  size_t getSize() const override {
    // Starts with 2 byte Hint field, followed by a null-terminated string,
    // ends with 0 or 1 byte padding.
    return RoundUpToAlignment(Name.size() + 3, 2);
  }

  void writeTo(uint8_t *Buf) override {
    write16le(Buf + FileOff, Hint);
    memcpy(Buf + FileOff + 2, Name.data(), Name.size());
  }

private:
  StringRef Name;
  uint16_t Hint;
};

// A chunk for the import descriptor table.
class LookupChunk : public Chunk {
public:
  explicit LookupChunk(Chunk *C) : HintName(C) {}
  size_t getSize() const override { return LookupChunkSize; }

  void writeTo(uint8_t *Buf) override {
    write32le(Buf + FileOff, HintName->getRVA());
  }

  Chunk *HintName;
};

// A chunk for the import descriptor table.
// This chunk represent import-by-ordinal symbols.
// See Microsoft PE/COFF spec 7.1. Import Header for details.
class OrdinalOnlyChunk : public Chunk {
public:
  explicit OrdinalOnlyChunk(uint16_t V) : Ordinal(V) {}
  size_t getSize() const override { return sizeof(uint64_t); }

  void writeTo(uint8_t *Buf) override {
    // An import-by-ordinal slot has MSB 1 to indicate that
    // this is import-by-ordinal (and not import-by-name).
    write64le(Buf + FileOff, (uint64_t(1) << 63) | Ordinal);
  }

  uint16_t Ordinal;
};

// A chunk for the import descriptor table.
class ImportDirectoryChunk : public Chunk {
public:
  explicit ImportDirectoryChunk(Chunk *N) : DLLName(N) {}
  size_t getSize() const override { return sizeof(ImportDirectoryTableEntry); }

  void writeTo(uint8_t *Buf) override {
    auto *E = (coff_import_directory_table_entry *)(Buf + FileOff);
    E->ImportLookupTableRVA = LookupTab->getRVA();
    E->NameRVA = DLLName->getRVA();
    E->ImportAddressTableRVA = AddressTab->getRVA();
  }

  Chunk *DLLName;
  Chunk *LookupTab;
  Chunk *AddressTab;
};

// A chunk representing null terminator in the import table.
// Contents of this chunk is always null bytes.
class NullChunk : public Chunk {
public:
  explicit NullChunk(size_t N) : Size(N) {}
  bool hasData() const override { return false; }
  size_t getSize() const override { return Size; }

private:
  size_t Size;
};

uint64_t IdataContents::getDirSize() {
  return Dirs.size() * sizeof(ImportDirectoryTableEntry);
}

uint64_t IdataContents::getIATSize() {
  return Addresses.size() * LookupChunkSize;
}

// Returns a list of .idata contents.
// See Microsoft PE/COFF spec 5.4 for details.
std::vector<Chunk *> IdataContents::getChunks() {
  create();
  std::vector<Chunk *> V;
  // The loader assumes a specific order of data.
  // Add each type in the correct order.
  for (std::unique_ptr<Chunk> &C : Dirs)
    V.push_back(C.get());
  for (std::unique_ptr<Chunk> &C : Lookups)
    V.push_back(C.get());
  for (std::unique_ptr<Chunk> &C : Addresses)
    V.push_back(C.get());
  for (std::unique_ptr<Chunk> &C : Hints)
    V.push_back(C.get());
  for (auto &P : DLLNames) {
    std::unique_ptr<Chunk> &C = P.second;
    V.push_back(C.get());
  }
  return V;
}

void IdataContents::create() {
  // Group DLL-imported symbols by DLL name because that's how
  // symbols are layed out in the import descriptor table.
  std::map<StringRef, std::vector<DefinedImportData *>> Map;
  for (DefinedImportData *Sym : Imports)
    Map[Sym->getDLLName()].push_back(Sym);

  // Create .idata contents for each DLL.
  for (auto &P : Map) {
    StringRef Name = P.first;
    std::vector<DefinedImportData *> &Syms = P.second;

    // Sort symbols by name for each group.
    std::sort(Syms.begin(), Syms.end(),
              [](DefinedImportData *A, DefinedImportData *B) {
                return A->getName() < B->getName();
              });

    // Create lookup and address tables. If they have external names,
    // we need to create HintName chunks to store the names.
    // If they don't (if they are import-by-ordinals), we store only
    // ordinal values to the table.
    size_t Base = Lookups.size();
    for (DefinedImportData *S : Syms) {
      uint16_t Ord = S->getOrdinal();
      if (S->getExternalName().empty()) {
        Lookups.push_back(make_unique<OrdinalOnlyChunk>(Ord));
        Addresses.push_back(make_unique<OrdinalOnlyChunk>(Ord));
        continue;
      }
      auto C = make_unique<HintNameChunk>(S->getExternalName(), Ord);
      Lookups.push_back(make_unique<LookupChunk>(C.get()));
      Addresses.push_back(make_unique<LookupChunk>(C.get()));
      Hints.push_back(std::move(C));
    }
    // Terminate with null values.
    Lookups.push_back(make_unique<NullChunk>(LookupChunkSize));
    Addresses.push_back(make_unique<NullChunk>(LookupChunkSize));

    for (int I = 0, E = Syms.size(); I < E; ++I)
      Syms[I]->setLocation(Addresses[Base + I].get());

    // Create the import table header.
    if (!DLLNames.count(Name))
      DLLNames[Name] = make_unique<StringChunk>(Name);
    auto Dir = make_unique<ImportDirectoryChunk>(DLLNames[Name].get());
    Dir->LookupTab = Lookups[Base].get();
    Dir->AddressTab = Addresses[Base].get();
    Dirs.push_back(std::move(Dir));
  }
  // Add null terminator.
  Dirs.push_back(make_unique<NullChunk>(sizeof(ImportDirectoryTableEntry)));
}

// Export table
// Read Microsoft PE/COFF spec 5.3 for details.

// A chunk for the export descriptor table.
class ExportDirectoryChunk : public Chunk {
public:
  ExportDirectoryChunk(int I, int J, Chunk *D, Chunk *A, Chunk *N, Chunk *O)
      : MaxOrdinal(I), NameTabSize(J), DLLName(D), AddressTab(A), NameTab(N),
        OrdinalTab(O) {}

  size_t getSize() const override {
    return sizeof(export_directory_table_entry);
  }

  void writeTo(uint8_t *Buf) override {
    auto *E = (export_directory_table_entry *)(Buf + FileOff);
    E->NameRVA = DLLName->getRVA();
    E->OrdinalBase = 0;
    E->AddressTableEntries = MaxOrdinal + 1;
    E->NumberOfNamePointers = NameTabSize;
    E->ExportAddressTableRVA = AddressTab->getRVA();
    E->NamePointerRVA = NameTab->getRVA();
    E->OrdinalTableRVA = OrdinalTab->getRVA();
  }

  uint16_t MaxOrdinal;
  uint16_t NameTabSize;
  Chunk *DLLName;
  Chunk *AddressTab;
  Chunk *NameTab;
  Chunk *OrdinalTab;
};

class AddressTableChunk : public Chunk {
public:
  explicit AddressTableChunk(size_t MaxOrdinal) : Size(MaxOrdinal + 1) {}
  size_t getSize() const override { return Size * 4; }

  void writeTo(uint8_t *Buf) override {
    for (Export &E : Config->Exports)
      write32le(Buf + FileOff + E.Ordinal * 4, E.Sym->getRVA());
  }

private:
  size_t Size;
};

class NamePointersChunk : public Chunk {
public:
  explicit NamePointersChunk(std::vector<Chunk *> &V) : Chunks(V) {}
  size_t getSize() const override { return Chunks.size() * 4; }

  void writeTo(uint8_t *Buf) override {
    uint8_t *P = Buf + FileOff;
    for (Chunk *C : Chunks) {
      write32le(P, C->getRVA());
      P += 4;
    }
  }

private:
  std::vector<Chunk *> Chunks;
};

class ExportOrdinalChunk : public Chunk {
public:
  explicit ExportOrdinalChunk(size_t I) : Size(I) {}
  size_t getSize() const override { return Size * 2; }

  void writeTo(uint8_t *Buf) override {
    uint8_t *P = Buf + FileOff;
    for (Export &E : Config->Exports) {
      if (E.Noname)
        continue;
      write16le(P, E.Ordinal);
      P += 2;
    }
  }

private:
  size_t Size;
};

EdataContents::EdataContents() {
  uint16_t MaxOrdinal = 0;
  for (Export &E : Config->Exports)
    MaxOrdinal = std::max(MaxOrdinal, E.Ordinal);

  auto *DLLName = new StringChunk(sys::path::filename(Config->OutputFile));
  auto *AddressTab = new AddressTableChunk(MaxOrdinal);
  std::vector<Chunk *> Names;
  for (Export &E : Config->Exports)
    if (!E.Noname)
      Names.push_back(new StringChunk(E.ExtName));
  auto *NameTab = new NamePointersChunk(Names);
  auto *OrdinalTab = new ExportOrdinalChunk(Names.size());
  auto *Dir = new ExportDirectoryChunk(MaxOrdinal, Names.size(), DLLName,
                                       AddressTab, NameTab, OrdinalTab);
  Chunks.push_back(std::unique_ptr<Chunk>(Dir));
  Chunks.push_back(std::unique_ptr<Chunk>(DLLName));
  Chunks.push_back(std::unique_ptr<Chunk>(AddressTab));
  Chunks.push_back(std::unique_ptr<Chunk>(NameTab));
  Chunks.push_back(std::unique_ptr<Chunk>(OrdinalTab));
  for (Chunk *C : Names)
    Chunks.push_back(std::unique_ptr<Chunk>(C));
}

} // namespace coff
} // namespace lld
