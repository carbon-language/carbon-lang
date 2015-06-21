//===- DLL.cpp ------------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines various types of chunks for the DLL import or export
// descriptor tables. They are inherently Windows-specific.
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
  void setAlign(size_t N) { Align = N; }

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

static std::map<StringRef, std::vector<DefinedImportData *>>
binImports(const std::vector<DefinedImportData *> &Imports) {
  // Group DLL-imported symbols by DLL name because that's how
  // symbols are layed out in the import descriptor table.
  std::map<StringRef, std::vector<DefinedImportData *>> M;
  for (DefinedImportData *Sym : Imports)
    M[Sym->getDLLName()].push_back(Sym);

  for (auto &P : M) {
    // Sort symbols by name for each group.
    std::vector<DefinedImportData *> &Syms = P.second;
    std::sort(Syms.begin(), Syms.end(),
              [](DefinedImportData *A, DefinedImportData *B) {
                return A->getName() < B->getName();
              });
  }
  return M;
}

void IdataContents::create() {
  std::map<StringRef, std::vector<DefinedImportData *>> Map =
      binImports(Imports);

  // Create .idata contents for each DLL.
  for (auto &P : Map) {
    StringRef Name = P.first;
    std::vector<DefinedImportData *> &Syms = P.second;

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
// See Microsoft PE/COFF spec 4.3 for details.

// A chunk for the delay import descriptor table etnry.
class DelayDirectoryChunk : public Chunk {
public:
  explicit DelayDirectoryChunk(Chunk *N) : DLLName(N) {}

  size_t getSize() const override {
    return sizeof(delay_import_directory_table_entry);
  }

  void writeTo(uint8_t *Buf) override {
    auto *E = (delay_import_directory_table_entry *)(Buf + FileOff);
    E->Name = DLLName->getRVA();
    E->ModuleHandle = ModuleHandle->getRVA();
    E->DelayImportAddressTable = AddressTab->getRVA();
    E->DelayImportNameTable = NameTab->getRVA();
  }

  Chunk *DLLName;
  Chunk *ModuleHandle;
  Chunk *AddressTab;
  Chunk *NameTab;
};

// Initial contents for delay-loaded functions.
// This code calls __delayLoadHerper2 function to resolve a symbol
// and then overwrites its jump table slot with the result
// for subsequent function calls.
static const uint8_t Thunk[] = {
    0x51,                               // push    rcx
    0x52,                               // push    rdx
    0x41, 0x50,                         // push    r8
    0x41, 0x51,                         // push    r9
    0x48, 0x83, 0xEC, 0x48,             // sub     rsp, 48h
    0x66, 0x0F, 0x7F, 0x04, 0x24,       // movdqa  xmmword ptr [rsp], xmm0
    0x66, 0x0F, 0x7F, 0x4C, 0x24, 0x10, // movdqa  xmmword ptr [rsp+10h], xmm1
    0x66, 0x0F, 0x7F, 0x54, 0x24, 0x20, // movdqa  xmmword ptr [rsp+20h], xmm2
    0x66, 0x0F, 0x7F, 0x5C, 0x24, 0x30, // movdqa  xmmword ptr [rsp+30h], xmm3
    0x48, 0x8D, 0x15, 0, 0, 0, 0,       // lea     rdx, [__imp_<FUNCNAME>]
    0x48, 0x8D, 0x0D, 0, 0, 0, 0,       // lea     rcx, [___DELAY_IMPORT_...]
    0xE8, 0, 0, 0, 0,                   // call    __delayLoadHelper2
    0x66, 0x0F, 0x6F, 0x04, 0x24,       // movdqa  xmm0, xmmword ptr [rsp]
    0x66, 0x0F, 0x6F, 0x4C, 0x24, 0x10, // movdqa  xmm1, xmmword ptr [rsp+10h]
    0x66, 0x0F, 0x6F, 0x54, 0x24, 0x20, // movdqa  xmm2, xmmword ptr [rsp+20h]
    0x66, 0x0F, 0x6F, 0x5C, 0x24, 0x30, // movdqa  xmm3, xmmword ptr [rsp+30h]
    0x48, 0x83, 0xC4, 0x48,             // add     rsp, 48h
    0x41, 0x59,                         // pop     r9
    0x41, 0x58,                         // pop     r8
    0x5A,                               // pop     rdx
    0x59,                               // pop     rcx
    0xFF, 0xE0,                         // jmp     rax
};

// A chunk for the delay import thunk.
class ThunkChunk : public Chunk {
public:
  ThunkChunk(Defined *I, Defined *H) : Imp(I), Helper(H) {}

  size_t getSize() const override { return sizeof(Thunk); }

  void writeTo(uint8_t *Buf) override {
    memcpy(Buf + FileOff, Thunk, sizeof(Thunk));
    write32le(Buf + FileOff + 36, Imp->getRVA());
    write32le(Buf + FileOff + 43, Desc->getRVA());
    write32le(Buf + FileOff + 48, Helper->getRVA());
  }

  Defined *Imp = nullptr;
  Chunk *Desc = nullptr;
  Defined *Helper = nullptr;
};

std::vector<Chunk *> DelayLoadContents::getChunks(Defined *H) {
  Helper = H;
  create();
  std::vector<Chunk *> V;
  for (std::unique_ptr<Chunk> &C : Dirs)
    V.push_back(C.get());
  for (std::unique_ptr<Chunk> &C : Addresses)
    V.push_back(C.get());
  for (std::unique_ptr<Chunk> &C : Names)
    V.push_back(C.get());
  for (std::unique_ptr<Chunk> &C : HintNames)
    V.push_back(C.get());
  for (auto &P : DLLNames) {
    std::unique_ptr<Chunk> &C = P.second;
    V.push_back(C.get());
  }
  return V;
}

uint64_t DelayLoadContents::getDirSize() {
  return Dirs.size() * sizeof(delay_import_directory_table_entry);
}

void DelayLoadContents::create() {
  std::map<StringRef, std::vector<DefinedImportData *>> Map =
      binImports(Imports);

  // Create .didat contents for each DLL.
  for (auto &P : Map) {
    StringRef Name = P.first;
    std::vector<DefinedImportData *> &Syms = P.second;

    size_t Base = Addresses.size();
    for (DefinedImportData *S : Syms) {
      auto T = make_unique<ThunkChunk>(S, Helper);
      auto A = make_unique<LookupChunk>(T.get());
      T->Desc = A.get();
      Addresses.push_back(std::move(A));
      Thunks.push_back(std::move(T));
      auto C =
          make_unique<HintNameChunk>(S->getExternalName(), S->getOrdinal());
      Names.push_back(make_unique<LookupChunk>(C.get()));
      HintNames.push_back(std::move(C));
    }
    // Terminate with null values.
    Addresses.push_back(make_unique<NullChunk>(LookupChunkSize));
    Names.push_back(make_unique<NullChunk>(LookupChunkSize));

    for (int I = 0, E = Syms.size(); I < E; ++I)
      Syms[I]->setLocation(Addresses[Base + I].get());
    auto *MH = new NullChunk(8);
    MH->setAlign(8);
    ModuleHandles.push_back(std::unique_ptr<Chunk>(MH));

    // Create the delay import table header.
    if (!DLLNames.count(Name))
      DLLNames[Name] = make_unique<StringChunk>(Name);
    auto Dir = make_unique<DelayDirectoryChunk>(DLLNames[Name].get());
    Dir->ModuleHandle = MH;
    Dir->AddressTab = Addresses[Base].get();
    Dir->NameTab = Names[Base].get();
    Dirs.push_back(std::move(Dir));
  }
  // Add null terminator.
  Dirs.push_back(
      make_unique<NullChunk>(sizeof(delay_import_directory_table_entry)));
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
