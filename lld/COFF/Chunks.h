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
#include "llvm/ADT/iterator_range.h"
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
class DefinedRegular;
class DefinedImportData;
class ObjectFile;
class OutputSection;

// A Chunk represents a chunk of data that will occupy space in the
// output (if the resolver chose that). It may or may not be backed by
// a section of an input file. It could be linker-created data, or
// doesn't even have actual data (if common or bss).
class Chunk {
public:
  enum Kind { SectionKind, OtherKind };
  Kind kind() const { return ChunkKind; }
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

  // An output section has pointers to chunks in the section, and each
  // chunk has a back pointer to an output section.
  void setOutputSection(OutputSection *O) { Out = O; }
  OutputSection *getOutputSection() { return Out; }

  // Windows-specific.
  // Collect all locations that contain absolute addresses for base relocations.
  virtual void getBaserels(std::vector<uint32_t> *Res, Defined *ImageBase) {}

  // Returns a human-readable name of this chunk. Chunks are unnamed chunks of
  // bytes, so this is used only for logging or debugging.
  virtual StringRef getDebugName() { return ""; }

protected:
  Chunk(Kind K = OtherKind) : ChunkKind(K) {}
  const Kind ChunkKind;

  // The RVA of this chunk in the output. The writer sets a value.
  uint64_t RVA = 0;

  // The offset from beginning of the output file. The writer sets a value.
  uint64_t FileOff = 0;

  // The output section for this chunk.
  OutputSection *Out = nullptr;

  // The alignment of this chunk. The writer uses the value.
  uint32_t Align = 1;
};

// A chunk corresponding a section of an input file.
class SectionChunk : public Chunk {
public:
  SectionChunk(ObjectFile *File, const coff_section *Header);
  static bool classof(const Chunk *C) { return C->kind() == SectionKind; }
  size_t getSize() const override { return Header->SizeOfRawData; }
  void writeTo(uint8_t *Buf) override;
  bool hasData() const override;
  uint32_t getPermissions() const override;
  StringRef getSectionName() const override { return SectionName; }
  void getBaserels(std::vector<uint32_t> *Res, Defined *ImageBase) override;
  bool isCOMDAT() const;

  // Called if the garbage collector decides to not include this chunk
  // in a final output. It's supposed to print out a log message to stdout.
  void printDiscardedMessage() const;

  // Adds COMDAT associative sections to this COMDAT section. A chunk
  // and its children are treated as a group by the garbage collector.
  void addAssociative(SectionChunk *Child);

  StringRef getDebugName() override;
  void setSymbol(DefinedRegular *S) { if (!Sym) Sym = S; }

  // Used by the garbage collector.
  bool isRoot() { return Root; }
  bool isLive() { return Live; }
  void markLive() { if (!Live) mark(); }

  // Used for ICF (Identical COMDAT Folding)
  void replaceWith(SectionChunk *Other);
  uint64_t getHash() const;
  bool equals(const SectionChunk *Other) const;

  // A pointer pointing to a replacement for this chunk.
  // Initially it points to "this" object. If this chunk is merged
  // with other chunk by ICF, it points to another chunk,
  // and this chunk is considrered as dead.
  SectionChunk *Ptr;

private:
  ArrayRef<uint8_t> getContents() const;

  // A file this chunk was created from.
  ObjectFile *File;

  const coff_section *Header;
  StringRef SectionName;
  std::vector<Chunk *> AssocChildren;
  llvm::iterator_range<const coff_relocation *> Relocs;
  size_t NumRelocs;

  // Used by the garbage collector.
  void mark();
  bool Live = false;
  bool Root;

  // Chunks are basically unnamed chunks of bytes.
  // Symbols are associated for debugging and logging purposs only.
  DefinedRegular *Sym = nullptr;
};

// A chunk for common symbols. Common chunks don't have actual data.
class CommonChunk : public Chunk {
public:
  CommonChunk(const COFFSymbolRef Sym);
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

static const uint8_t ImportThunkData[] = {
    0xff, 0x25, 0x00, 0x00, 0x00, 0x00, // JMP *0x0
};

// Windows-specific.
// A chunk for DLL import jump table entry. In a final output, it's
// contents will be a JMP instruction to some __imp_ symbol.
class ImportThunkChunk : public Chunk {
public:
  explicit ImportThunkChunk(Defined *ImpSymbol);
  size_t getSize() const override { return sizeof(ImportThunkData); }
  void writeTo(uint8_t *Buf) override;

private:
  Defined *ImpSymbol;
};

// Windows-specific.
// See comments for DefinedLocalImport class.
class LocalImportChunk : public Chunk {
public:
  explicit LocalImportChunk(Defined *S) : Sym(S) {}
  size_t getSize() const override { return 4; }
  void writeTo(uint8_t *Buf) override;

private:
  Defined *Sym;
};

// Windows-specific.
// This class represents a block in .reloc section.
// See the PE/COFF spec 5.6 for details.
class BaserelChunk : public Chunk {
public:
  BaserelChunk(uint32_t Page, uint32_t *Begin, uint32_t *End);
  size_t getSize() const override { return Data.size(); }
  void writeTo(uint8_t *Buf) override;

private:
  std::vector<uint8_t> Data;
};

} // namespace coff
} // namespace lld

#endif
