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
#include "llvm/Support/Allocator.h"
#include <unordered_map>

namespace llvm {
struct LTOCodeGenerator;
}

namespace lld {
namespace elfv2 {

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

  std::error_code addFile(std::unique_ptr<InputFile> File);

  // Print an error message on undefined symbols.
  bool reportRemainingUndefines();

  // Returns a list of chunks of selected symbols.
  std::vector<Chunk *> getChunks();

  // Returns a symbol for a given name. It's not guaranteed that the
  // returned symbol actually has the same name (because of various
  // mechanisms to allow aliases, a name can be resolved to a
  // different symbol). Returns a nullptr if not found.
  Defined *find(StringRef Name);

  // Dump contents of the symbol table to stderr.
  void dump();

  // Build an ELF object representing the combined contents of BitcodeFiles
  // and add it to the symbol table. Called after all files are added and
  // before the writer writes results to a file.
  std::error_code addCombinedLTOObject();

  // The writer needs to infer the machine type from the object files.
  std::vector<std::unique_ptr<ObjectFile<ELFT>>> ObjectFiles;

  // Creates an Undefined symbol for a given name.
  std::error_code addUndefined(StringRef Name);

  // Rename From -> To in the symbol table.
  std::error_code rename(StringRef From, StringRef To);

private:
  std::error_code addObject(ObjectFile<ELFT> *File);
  std::error_code addArchive(ArchiveFile *File);
  std::error_code addBitcode(BitcodeFile *File);

  std::error_code resolve(SymbolBody *Body);
  std::error_code addMemberFile(Lazy *Body);
  ErrorOr<ObjectFile<ELFT> *> createLTOObject(llvm::LTOCodeGenerator *CG);

  std::unordered_map<StringRef, Symbol *> Symtab;
  std::vector<std::unique_ptr<ArchiveFile>> ArchiveFiles;
  std::vector<std::unique_ptr<BitcodeFile>> BitcodeFiles;
  std::unique_ptr<MemoryBuffer> LTOMB;
  llvm::BumpPtrAllocator Alloc;
};

} // namespace elfv2
} // namespace lld

#endif
