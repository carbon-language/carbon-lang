//===- SymbolTable.h ------------------------------------------------------===//
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
class Defined;
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
template <class ELFT> class SymbolTable {
public:
  SymbolTable();

  void addFile(std::unique_ptr<InputFile> File);

  // Print an error message on undefined symbols.
  void reportRemainingUndefines();

  // The writer needs to infer the machine type from the object files.
  std::vector<std::unique_ptr<ObjectFile<ELFT>>> ObjectFiles;

private:
  void addObject(ObjectFile<ELFT> *File);

  void resolve(SymbolBody *Body);

  llvm::DenseMap<StringRef, Symbol *> Symtab;
  llvm::BumpPtrAllocator Alloc;
};

} // namespace elf2
} // namespace lld

#endif
