//===- Chunks.h -----------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_CHUNKS_H
#define LLD_COFF_CHUNKS_H

#include "lld/Core/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Object/COFF.h"
#include <map>
#include <vector>

namespace lld {
namespace coff {

using llvm::COFF::ImportDirectoryTableEntry;
using llvm::object::COFFSymbolRef;
using llvm::object::SectionRef;
using llvm::object::coff_relocation;
using llvm::object::coff_section;
using llvm::sys::fs::file_magic;

class Defined;
class DefinedImportData;
class ObjectFile;
class OutputSection;

// A Chunk represents a chunk of data that will occupy space in the
// output (if the resolver chose that). It may or may not be backed by
// a section of an input file. It could be linker-created data, or
// doesn't even have actual data (if common or bss).
class Chunk {
public:
  virtual ~Chunk() = default;

  // Returns the size of this chunk (even if this is a common or BSS.)
  virtual size_t getSize() const = 0;

  // Write this chunk to a mmap'ed file, assuming Buf is pointing to
  // beginning of the file. Because this function may use RVA values
  // of other chunks for relocations, you need to set them properly
  // before calling this function.
  virtual void writeTo(uint8_t *Buf) {}

  // The writer sets and uses the addresses.
  uint64_t getRVA() { return RVA; }
  uint64_t getFileOff() { return FileOff; }
  uint32_t getAlign() { return Align; }
  void setRVA(uint64_t V) { RVA = V; }
  void setFileOff(uint64_t V) { FileOff = V; }

  // Returns true if this has non-zero data. BSS chunks return
  // false. If false is returned, the space occupied by this chunk
  // will be filled with zeros.
  virtual bool hasData() const { return true; }

  // Returns readable/writable/executable bits.
  virtual uint32_t getPermissions() const { return 0; }

  // Returns the section name if this is a section chunk.
  // It is illegal to call this function on non-section chunks.
  virtual StringRef getSectionName() const {
    llvm_unreachable("unimplemented getSectionName");
  }

  // Called if the garbage collector decides to not include this chunk
  // in a final output. It's supposed to print out a log message to stderr.
  // It is illegal to call this function on non-section chunks because
  // only section chunks are subject of garbage collection.
  virtual void printDiscardedMessage() {
    llvm_unreachable("unimplemented printDiscardedMessage");
  }

  // Returns true if this is a COMDAT section. Usually, it is an error
  // if there are more than one defined symbols having the same name,
  // but symbols at begining of COMDAT sections allowed to duplicate.
  virtual bool isCOMDAT() const { return false; }

  // Used by the garbage collector.
  virtual bool isRoot() { return false; }
  virtual bool isLive() { return true; }
  virtual void markLive() {}

  // An output section has pointers to chunks in the section, and each
  // chunk has a back pointer to an output section.
  void setOutputSection(OutputSection *O) { Out = O; }
  OutputSection *getOutputSection() { return Out; }

protected:
  // The RVA of this chunk in the output. The writer sets a value.
  uint64_t RVA = 0;

  // The offset from beginning of the output file. The writer sets a value.
  uint64_t FileOff = 0;

  // The alignment of this chunk. The writer uses the value.
  uint32_t Align = 1;

  // The output section for this chunk.
  OutputSection *Out = nullptr;
};

// A chunk corresponding a section of an input file.
class SectionChunk : public Chunk {
public:
  SectionChunk(ObjectFile *File, const coff_section *Header,
               uint32_t SectionIndex);
  size_t getSize() const override { return Header->SizeOfRawData; }
  void writeTo(uint8_t *Buf) override;
  bool hasData() const override;
  uint32_t getPermissions() const override;
  StringRef getSectionName() const override { return SectionName; }
  void printDiscardedMessage() override;
  bool isCOMDAT() const override;

  bool isRoot() override;
  void markLive() override;
  bool isLive() override { return Live; }

  // Adds COMDAT associative sections to this COMDAT section. A chunk
  // and its children are treated as a group by the garbage collector.
  void addAssociative(SectionChunk *Child);

private:
  SectionRef getSectionRef();
  void applyReloc(uint8_t *Buf, const coff_relocation *Rel);

  // A file this chunk was created from.
  ObjectFile *File;

  const coff_section *Header;
  uint32_t SectionIndex;
  StringRef SectionName;
  bool Live = false;
  std::vector<Chunk *> AssocChildren;
  bool IsAssocChild = false;
};

// A chunk for common symbols. Common chunks don't have actual data.
class CommonChunk : public Chunk {
public:
  CommonChunk(const COFFSymbolRef S) : Sym(S) {}
  size_t getSize() const override { return Sym.getValue(); }
  bool hasData() const override { return false; }
  uint32_t getPermissions() const override;
  StringRef getSectionName() const override { return ".bss"; }

private:
  const COFFSymbolRef Sym;
};

// A chunk for linker-created strings.
class StringChunk : public Chunk {
public:
  explicit StringChunk(StringRef S) : Str(S) {}
  size_t getSize() const override { return Str.size() + 1; }
  void writeTo(uint8_t *Buf) override;

private:
  StringRef Str;
};

// All chunks below are for the DLL import descriptor table and
// Windows-specific. You may need to read the Microsoft PE/COFF spec
// to understand details about the data structures.
// If you are not particularly interested, you can skip them and
// still be able to understand the rest of the linker.

static const uint8_t ImportThunkData[] = {
    0xff, 0x25, 0x00, 0x00, 0x00, 0x00, // JMP *0x0
};

// A chunk for DLL import jump table entry. In a final output, it's
// contents will be a JMP instruction to some __imp_ symbol.
class ImportThunkChunk : public Chunk {
public:
  explicit ImportThunkChunk(Defined *S) : ImpSymbol(S) {}
  size_t getSize() const override { return sizeof(ImportThunkData); }
  void writeTo(uint8_t *Buf) override;

private:
  Defined *ImpSymbol;
};

// A chunk for the import descriptor table.
class HintNameChunk : public Chunk {
public:
  HintNameChunk(StringRef N, uint16_t H) : Name(N), Hint(H) {}
  size_t getSize() const override;
  void writeTo(uint8_t *Buf) override;

private:
  StringRef Name;
  uint16_t Hint;
};

// A chunk for the import descriptor table.
class LookupChunk : public Chunk {
public:
  explicit LookupChunk(Chunk *C) : HintName(C) {}
  size_t getSize() const override { return Size; }
  void writeTo(uint8_t *Buf) override;
  static const size_t Size;
  Chunk *HintName;
};

// A chunk for the import descriptor table.
// This chunk represent import-by-ordinal symbols.
// See the Microsoft PE/COFF spec 7.1. Import Header for details.
class OrdinalOnlyChunk : public Chunk {
public:
  explicit OrdinalOnlyChunk(uint16_t V) : Ordinal(V) {}
  size_t getSize() const override { return sizeof(uint64_t); }
  void writeTo(uint8_t *Buf) override;
  uint16_t Ordinal;
};

// A chunk for the import descriptor table.
class DirectoryChunk : public Chunk {
public:
  explicit DirectoryChunk(Chunk *N) : DLLName(N) {}
  size_t getSize() const override { return Size; }
  void writeTo(uint8_t *Buf) override;
  static const size_t Size;
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

// IdataContents creates all chunks for the .idata section.
// You are supposed to call add() to add symbols and then
// call getChunks() to get a list of chunks.
class IdataContents {
public:
  void add(DefinedImportData *Sym) { Imports.push_back(Sym); }
  std::vector<Chunk *> getChunks();

  uint64_t getDirRVA() { return Dirs[0]->getRVA(); }
  uint64_t getDirSize() { return Dirs.size() * DirectoryChunk::Size; }
  uint64_t getIATRVA() { return Addresses[0]->getRVA(); }
  uint64_t getIATSize() { return Addresses.size() * LookupChunk::Size; }

private:
  void create();

  std::vector<DefinedImportData *> Imports;
  std::vector<std::unique_ptr<Chunk>> Dirs;
  std::vector<std::unique_ptr<Chunk>> Lookups;
  std::vector<std::unique_ptr<Chunk>> Addresses;
  std::vector<std::unique_ptr<Chunk>> Hints;
  std::map<StringRef, std::unique_ptr<Chunk>> DLLNames;
};

} // namespace coff
} // namespace lld

#endif
