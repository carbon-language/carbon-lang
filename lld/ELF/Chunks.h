//===- Chunks.h -----------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_CHUNKS_H
#define LLD_ELF_CHUNKS_H

#include "lld/Core/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Object/ELF.h"
#include <map>
#include <vector>

namespace lld {
namespace elfv2 {

class Defined;
template <class ELFT> class ObjectFile;
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
  // beginning of the file. Because this function may use VA values
  // of other chunks for relocations, you need to set them properly
  // before calling this function.
  virtual void writeTo(uint8_t *Buf) {}

  // The writer sets and uses the addresses.
  uint64_t getVA() { return VA; }
  uint64_t getFileOff() { return FileOff; }
  uint32_t getAlign() { return Align; }
  void setVA(uint64_t V) { VA = V; }
  void setFileOff(uint64_t V) { FileOff = V; }

  // Returns true if this has non-zero data. BSS chunks return
  // false. If false is returned, the space occupied by this chunk
  // will be filled with zeros.
  virtual bool hasData() const { return true; }

  // Returns readable/writable/executable bits.
  virtual uint32_t getFlags() const { return 0; }

  // Returns the section name if this is a section chunk.
  // It is illegal to call this function on non-section chunks.
  virtual StringRef getSectionName() const {
    llvm_unreachable("unimplemented getSectionName");
  }

  // Called if the garbage collector decides to not include this chunk
  // in a final output. It's supposed to print out a log message to stdout.
  // It is illegal to call this function on non-section chunks because
  // only section chunks are subject of garbage collection.
  virtual void printDiscardedMessage() {
    llvm_unreachable("unimplemented printDiscardedMessage");
  }

  // Used by the garbage collector.
  bool isRoot() { return Root; }
  bool isLive() { return Live; }
  void markLive() {
    if (!Live)
      mark();
  }

  // An output section has pointers to chunks in the section, and each
  // chunk has a back pointer to an output section.
  void setOutputSection(OutputSection *O) { Out = O; }
  OutputSection *getOutputSection() { return Out; }

protected:
  // The VA of this chunk in the output. The writer sets a value.
  uint64_t VA = 0;

  // The offset from beginning of the output file. The writer sets a value.
  uint64_t FileOff = 0;

  // The output section for this chunk.
  OutputSection *Out = nullptr;

  // The alignment of this chunk. The writer uses the value.
  uint32_t Align = 1;

  // Used by the garbage collector.
  virtual void mark() {}
  bool Live = true;
  bool Root = false;
};

// A chunk corresponding a section of an input file.
template <class ELFT> class SectionChunk : public Chunk {
  typedef llvm::object::Elf_Shdr_Impl<ELFT> Elf_Shdr;
  typedef llvm::object::Elf_Rel_Impl<ELFT, true> Elf_Rela;
  typedef llvm::object::Elf_Rel_Impl<ELFT, false> Elf_Rel;

public:
  SectionChunk(ObjectFile<ELFT> *File, const Elf_Shdr *Header,
               uint32_t SectionIndex);
  size_t getSize() const override { return Header->sh_size; }
  void writeTo(uint8_t *Buf) override;
  bool hasData() const override;
  uint32_t getFlags() const override;
  StringRef getSectionName() const override { return SectionName; }
  void printDiscardedMessage() override;

private:
  void mark() override;
  const Elf_Shdr *getSectionHdr();
  void applyReloc(uint8_t *Buf, const Elf_Rela *Rel);
  void applyReloc(uint8_t *Buf, const Elf_Rel *Rel);

  // A file this chunk was created from.
  ObjectFile<ELFT> *File;

  const Elf_Shdr *Header;
  uint32_t SectionIndex;
  StringRef SectionName;
};

// A chunk for common symbols. Common chunks don't have actual data.
template <class ELFT> class CommonChunk : public Chunk {
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;

public:
  CommonChunk(const Elf_Sym *Sym);
  size_t getSize() const override { return Sym->getValue(); }
  bool hasData() const override { return false; }
  uint32_t getFlags() const override;
  StringRef getSectionName() const override { return ".bss"; }

private:
  const Elf_Sym *Sym;
};

} // namespace elfv2
} // namespace lld

#endif
