//===- Writer.h -----------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_WRITER_H
#define LLD_COFF_WRITER_H

#include "DLL.h"
#include "InputFiles.h"
#include "SymbolTable.h"
#include "llvm/Support/FileOutputBuffer.h"
#include <memory>
#include <vector>

namespace lld {
namespace coff {

// Mask for section types (code, data, bss, disacardable, etc.)
// and permissions (writable, readable or executable).
const uint32_t PermMask = 0xFF0000F0;

// Implemented in ICF.cpp.
void doICF(const std::vector<Chunk *> &Chunks);

// OutputSection represents a section in an output file. It's a
// container of chunks. OutputSection and Chunk are 1:N relationship.
// Chunks cannot belong to more than one OutputSections. The writer
// creates multiple OutputSections and assign them unique,
// non-overlapping file offsets and RVAs.
class OutputSection {
public:
  OutputSection(StringRef N, uint32_t SI)
      : Name(N), SectionIndex(SI), Header({}) {}
  void setRVA(uint64_t);
  void setFileOffset(uint64_t);
  void addChunk(Chunk *C);
  StringRef getName() { return Name; }
  uint64_t getSectionIndex() { return SectionIndex; }
  std::vector<Chunk *> &getChunks() { return Chunks; }
  void addPermissions(uint32_t C);
  uint32_t getPermissions() { return Header.Characteristics & PermMask; }
  uint32_t getCharacteristics() { return Header.Characteristics; }
  uint64_t getRVA() { return Header.VirtualAddress; }
  uint64_t getFileOff() { return Header.PointerToRawData; }
  void writeHeaderTo(uint8_t *Buf);

  // Returns the size of this section in an executable memory image.
  // This may be smaller than the raw size (the raw size is multiple
  // of disk sector size, so there may be padding at end), or may be
  // larger (if that's the case, the loader reserves spaces after end
  // of raw data).
  uint64_t getVirtualSize() { return Header.VirtualSize; }

  // Returns the size of the section in the output file.
  uint64_t getRawSize() { return Header.SizeOfRawData; }

  // Set offset into the string table storing this section name.
  // Used only when the name is longer than 8 bytes.
  void setStringTableOff(uint32_t V) { StringTableOff = V; }

private:
  StringRef Name;
  uint32_t SectionIndex;
  coff_section Header;
  uint32_t StringTableOff = 0;
  std::vector<Chunk *> Chunks;
};

// The writer writes a SymbolTable result to a file.
class Writer {
public:
  explicit Writer(SymbolTable *T) : Symtab(T) {}
  std::error_code write(StringRef Path);

private:
  void markLive();
  void dedupCOMDATs();
  void createSections();
  void createImportTables();
  void createExportTable();
  void assignAddresses();
  void removeEmptySections();
  std::error_code openFile(StringRef OutputPath);
  void writeHeader();
  void writeSections();
  void sortExceptionTable();
  void applyRelocations();

  OutputSection *findSection(StringRef Name);
  OutputSection *createSection(StringRef Name);
  void addBaserels(OutputSection *Dest);
  void addBaserelBlocks(OutputSection *Dest, std::vector<uint32_t> &V);

  uint32_t getSizeOfInitializedData();
  std::map<StringRef, std::vector<DefinedImportData *>> binImports();

  SymbolTable *Symtab;
  std::unique_ptr<llvm::FileOutputBuffer> Buffer;
  llvm::SpecificBumpPtrAllocator<OutputSection> CAlloc;
  llvm::SpecificBumpPtrAllocator<BaserelChunk> BAlloc;
  std::vector<OutputSection *> OutputSections;
  IdataContents Idata;
  DelayLoadContents DelayIdata;
  EdataContents Edata;

  uint64_t FileSize;
  uint64_t SizeOfImage;
  uint64_t SizeOfHeaders;

  std::vector<std::unique_ptr<Chunk>> Chunks;
};

} // namespace coff
} // namespace lld

#endif
