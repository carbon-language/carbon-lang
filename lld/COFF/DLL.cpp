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

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::COFF;
using llvm::RoundUpToAlignment;

const size_t LookupChunkSize = sizeof(uint64_t);
const size_t DirectoryChunkSize = sizeof(ImportDirectoryTableEntry);

namespace lld {
namespace coff {

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
class DirectoryChunk : public Chunk {
public:
  explicit DirectoryChunk(Chunk *N) : DLLName(N) {}
  size_t getSize() const override { return DirectoryChunkSize; }

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
  return Dirs.size() * DirectoryChunkSize;
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
    Lookups.push_back(make_unique<NullChunk>(sizeof(uint64_t)));
    Addresses.push_back(make_unique<NullChunk>(sizeof(uint64_t)));

    for (int I = 0, E = Syms.size(); I < E; ++I)
      Syms[I]->setLocation(Addresses[Base + I].get());

    // Create the import table header.
    if (!DLLNames.count(Name))
      DLLNames[Name] = make_unique<StringChunk>(Name);
    auto Dir = make_unique<DirectoryChunk>(DLLNames[Name].get());
    Dir->LookupTab = Lookups[Base].get();
    Dir->AddressTab = Addresses[Base].get();
    Dirs.push_back(std::move(Dir));
  }
  // Add null terminator.
  Dirs.push_back(make_unique<NullChunk>(DirectoryChunkSize));
}

} // namespace coff
} // namespace lld
