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
#include "llvm/Object/ELF.h"

namespace lld {
namespace elf2 {

class Defined;
template <class ELFT> class ObjectFile;
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
  // beginning of the file. Because this function may use VA values
  // of other chunks for relocations, you need to set them properly
  // before calling this function.
  virtual void writeTo(uint8_t *Buf) = 0;

  // The writer sets and uses the addresses.
  uint64_t getVA() { return VA; }
  uint64_t getFileOff() { return FileOff; }
  uint32_t getAlign() { return Align; }
  void setVA(uint64_t V) { VA = V; }
  void setFileOff(uint64_t V) { FileOff = V; }

  // Returns the section name if this is a section chunk.
  // It is illegal to call this function on non-section chunks.
  virtual StringRef getSectionName() const = 0;

  // An output section has pointers to chunks in the section, and each
  // chunk has a back pointer to an output section.
  void setOutputSection(OutputSection *O) { Out = O; }
  OutputSection *getOutputSection() { return Out; }

protected:
  Chunk(Kind K = OtherKind) : ChunkKind(K) {}
  const Kind ChunkKind;

  // The VA of this chunk in the output. The writer sets a value.
  uint64_t VA = 0;

  // The offset from beginning of the output file. The writer sets a value.
  uint64_t FileOff = 0;

  // The output section for this chunk.
  OutputSection *Out = nullptr;

  // The alignment of this chunk. The writer uses the value.
  uint32_t Align = 1;
};

// A chunk corresponding a section of an input file.
template <class ELFT> class SectionChunk : public Chunk {
  typedef llvm::object::Elf_Shdr_Impl<ELFT> Elf_Shdr;
  typedef llvm::object::Elf_Rel_Impl<ELFT, true> Elf_Rela;
  typedef llvm::object::Elf_Rel_Impl<ELFT, false> Elf_Rel;

public:
  SectionChunk(llvm::object::ELFFile<ELFT> *Obj, const Elf_Shdr *Header);
  size_t getSize() const override { return Header->sh_size; }
  void writeTo(uint8_t *Buf) override;
  StringRef getSectionName() const override { return SectionName; }
  const Elf_Shdr *getSectionHdr() const { return Header; }

  static bool classof(const Chunk *C) {
    return C->kind() == SectionKind;
  }

private:
  // A file this chunk was created from.
  llvm::object::ELFFile<ELFT> *Obj;

  const Elf_Shdr *Header;
  StringRef SectionName;
};

} // namespace elf2
} // namespace lld

#endif
