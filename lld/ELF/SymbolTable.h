//===- SymbolTable.h --------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_SYMBOL_TABLE_H
#define LLD_ELF_SYMBOL_TABLE_H

#include "InputFiles.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"

namespace lld {
namespace elf2 {
struct Symbol;

// SymbolTable is a bucket of all known symbols, including defined,
// undefined, or lazy symbols (the last one is symbols in archive
// files whose archive members are not yet loaded).
//
// We put all symbols of all files to a SymbolTable, and the
// SymbolTable selects the "best" symbols if there are name
// conflicts. For example, obviously, a defined symbol is better than
// an undefined symbol. Or, if there's a conflict between a lazy and a
// undefined, it'll read an archive member to read a real definition
// to replace the lazy symbol. The logic is implemented in resolve().
class SymbolTable {
public:
  SymbolTable();

  void addFile(std::unique_ptr<InputFile> File);

  const ELFFileBase *getFirstELF() const {
    if (!ObjectFiles.empty())
      return ObjectFiles[0].get();
    if (!SharedFiles.empty())
      return SharedFiles[0].get();
    return nullptr;
  }

  const llvm::DenseMap<StringRef, Symbol *> &getSymbols() const {
    return Symtab;
  }

  const std::vector<std::unique_ptr<ObjectFileBase>> &getObjectFiles() const {
    return ObjectFiles;
  }

  const std::vector<std::unique_ptr<SharedFileBase>> &getSharedFiles() const {
    return SharedFiles;
  }

  SymbolBody *getEntrySym() const {
    if (!EntrySym)
      return nullptr;
    return EntrySym->getReplacement();
  }

private:
  Symbol *insert(SymbolBody *New);
  template <class ELFT> void addELFFile(ELFFileBase *File);
  void addELFFile(ELFFileBase *File);
  void addLazy(Lazy *New);
  void addMemberFile(Lazy *Body);

  template <class ELFT> void init();
  template <class ELFT> void resolve(SymbolBody *Body);

  std::vector<std::unique_ptr<ArchiveFile>> ArchiveFiles;

  llvm::DenseMap<StringRef, Symbol *> Symtab;
  llvm::BumpPtrAllocator Alloc;

  // The writer needs to infer the machine type from the object files.
  std::vector<std::unique_ptr<ObjectFileBase>> ObjectFiles;

  std::vector<std::unique_ptr<SharedFileBase>> SharedFiles;

  SymbolBody *EntrySym = nullptr;
};

} // namespace elf2
} // namespace lld

#endif
