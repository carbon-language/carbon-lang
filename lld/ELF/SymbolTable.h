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
#include "llvm/ADT/MapVector.h"

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

  bool shouldUseRela() const;

  const llvm::MapVector<StringRef, Symbol *> &getSymbols() const {
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
    return EntrySym->repl();
  }

  void addUndefinedSym(StringRef Name);

  template <class ELFT>
  void addSyntheticSym(StringRef Name, OutputSection<ELFT> &Section,
                       typename llvm::object::ELFFile<ELFT>::uintX_t Value);

  template <class ELFT> void addIgnoredSym(StringRef Name);

private:
  Symbol *insert(SymbolBody *New);
  template <class ELFT> void addELFFile(ELFFileBase *File);
  void addELFFile(ELFFileBase *File);
  void addLazy(Lazy *New);
  void addMemberFile(Lazy *Body);
  template <class ELFT> void addUndefinedSym(StringRef Name);

  template <class ELFT> void init(uint16_t EMachine);
  template <class ELFT> void resolve(SymbolBody *Body);
  template <class ELFT>
  void reportConflict(const SymbolBody &Old, const SymbolBody &New);

  std::vector<std::unique_ptr<ArchiveFile>> ArchiveFiles;

  // The order the global symbols are in is not defined. We can use an arbitrary
  // order, but it has to be reproducible. That is true even when cross linking.
  // The default hashing of StringRef produces different results on 32 and 64
  // bit systems so we use a MapVector. That is arbitrary, deterministic but
  // a bit inefficient.
  // FIXME: Experiment with passing in a custom hashing or sorting the symbols
  // once symbol resolution is finished.
  llvm::MapVector<StringRef, Symbol *> Symtab;
  llvm::BumpPtrAllocator Alloc;

  // The writer needs to infer the machine type from the object files.
  std::vector<std::unique_ptr<ObjectFileBase>> ObjectFiles;

  std::vector<std::unique_ptr<SharedFileBase>> SharedFiles;
  llvm::DenseSet<StringRef> IncludedSoNames;

  SymbolBody *EntrySym = nullptr;
};

} // namespace elf2
} // namespace lld

#endif
