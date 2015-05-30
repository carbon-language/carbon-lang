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

#include "InputFiles.h"
#include "SymbolTable.h"
#include "llvm/Support/FileOutputBuffer.h"
#include <memory>
#include <vector>

namespace lld {
namespace coff {

// Mask for section types (code, data or bss) and permissions
// (writable, readable or executable).
const uint32_t PermMask = 0xF00000F0;

// OutputSection represents a section in an output file. It's a
// container of chunks. OutputSection and Chunk are 1:N relationship.
// Chunks cannot belong to more than one OutputSections. The writer
// creates multiple OutputSections and assign them unique,
// non-overlapping file offsets and RVAs.
class OutputSection {
public:
  OutputSection(StringRef Name, uint32_t SectionIndex);
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
  void writeHeader(uint8_t *Buf);

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
  coff_section Header;
  StringRef Name;
  uint32_t SectionIndex;
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
  void createSections();
  void createImportTables();
  void assignAddresses();
  void removeEmptySections();
  std::error_code openFile(StringRef OutputPath);
  void writeHeader();
  void writeSections();
  void applyRelocations();

  OutputSection *findSection(StringRef Name);
  OutputSection *createSection(StringRef Name);

  uint32_t getSizeOfInitializedData();
  std::map<StringRef, std::vector<DefinedImportData *>> binImports();

  SymbolTable *Symtab;
  std::unique_ptr<llvm::FileOutputBuffer> Buffer;
  std::vector<std::unique_ptr<OutputSection>> OutputSections;
  Chunk *ImportAddressTable = nullptr;
  uint32_t ImportDirectoryTableSize = 0;
  uint32_t ImportAddressTableSize = 0;

  Defined *Entry;
  uint64_t FileSize;
  uint64_t SizeOfImage;
  uint64_t SizeOfHeaders;

  std::vector<std::unique_ptr<Chunk>> Chunks;
};

} // namespace pecoff
} // namespace lld

#endif
