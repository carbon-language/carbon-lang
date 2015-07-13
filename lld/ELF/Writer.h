//===- Writer.h -----------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_WRITER_H
#define LLD_ELF_WRITER_H

#include "InputFiles.h"
#include "SymbolTable.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/FileOutputBuffer.h"
#include <memory>
#include <vector>

namespace lld {
namespace elfv2 {

// OutputSection represents a section in an output file. It's a
// container of chunks. OutputSection and Chunk are 1:N relationship.
// Chunks cannot belong to more than one OutputSections. The writer
// creates multiple OutputSections and assign them unique,
// non-overlapping file offsets and VAs.
class OutputSection {
public:
  OutputSection(StringRef N, uint32_t SI)
      : Name(N), SectionIndex(SI), Header({}) {}
  void setVA(uint64_t);
  void setFileOffset(uint64_t);
  void addChunk(Chunk *C);
  StringRef getName() { return Name; }
  uint64_t getSectionIndex() { return SectionIndex; }
  std::vector<Chunk *> &getChunks() { return Chunks; }
  void addPermissions(uint32_t C);
  uint32_t getPermissions() { return 0; }
  uint64_t getVA() { return Header.sh_addr; }
  uint64_t getFileOff() { return Header.sh_offset; }
  void writeHeaderTo(uint8_t *Buf);

  // Returns the size of the section in the output file.
  uint64_t getSize() { return Header.sh_size; }

  // Set offset into the string table storing this section name.
  // Used only when the name is longer than 8 bytes.
  void setStringTableOff(uint32_t V) { StringTableOff = V; }

private:
  StringRef Name;
  uint32_t SectionIndex;
  llvm::ELF::Elf64_Shdr Header;
  uint32_t StringTableOff = 0;
  std::vector<Chunk *> Chunks;
};

// The writer writes a SymbolTable result to a file.
template <class ELFT> class Writer {
public:
  explicit Writer(SymbolTable<ELFT> *T) : Symtab(T) {}
  std::error_code write(StringRef Path);

private:
  void markLive();
  void createSections();
  void assignAddresses();
  void removeEmptySections();
  std::error_code openFile(StringRef OutputPath);
  void writeHeader();
  void writeSections();

  OutputSection *findSection(StringRef Name);

  SymbolTable<ELFT> *Symtab;
  std::unique_ptr<llvm::FileOutputBuffer> Buffer;
  llvm::SpecificBumpPtrAllocator<OutputSection> CAlloc;
  std::vector<OutputSection *> OutputSections;

  uint64_t FileSize;
  uint64_t SizeOfImage;
  uint64_t SizeOfHeaders;

  std::vector<std::unique_ptr<Chunk>> Chunks;
};

} // namespace elfv2
} // namespace lld

#endif
