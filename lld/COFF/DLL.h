//===- DLL.h -------------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_DLL_H
#define LLD_COFF_DLL_H

#include "Chunks.h"
#include "Symbols.h"

namespace lld {
namespace coff {

// Windows-specific.
// IdataContents creates all chunks for the .idata section.
// You are supposed to call add() to add symbols and then
// call getChunks() to get a list of chunks.
class IdataContents {
public:
  void add(DefinedImportData *Sym) { Imports.push_back(Sym); }
  std::vector<Chunk *> getChunks();

  uint64_t getDirRVA() { return Dirs[0]->getRVA(); }
  uint64_t getDirSize();
  uint64_t getIATRVA() { return Addresses[0]->getRVA(); }
  uint64_t getIATSize();

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
