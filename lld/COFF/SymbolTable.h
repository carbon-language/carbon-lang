//===- SymbolTable.h ------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_SYMBOL_TABLE_H
#define LLD_COFF_SYMBOL_TABLE_H

#include "InputFiles.h"
#include "Memory.h"
#include "llvm/Support/Allocator.h"
#include <unordered_map>

namespace lld {
namespace coff {

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

  // Windows specific -- `main` is not the only main function in Windows.
  // You can choose one from these four -- {w,}{WinMain,main}.
  // There are four different entry point functions for them,
  // {w,}{WinMain,main}CRTStartup, respectively. The linker needs to
  // choose the right one depending on which `main` function is defined.
  // This function looks up the symbol table and resolve corresponding
  // entry point name.
  ErrorOr<StringRef> findDefaultEntry();

  // Dump contents of the symbol table to stderr.
  void dump();

  // Build a COFF object representing the combined contents of BitcodeFiles
  // and add it to the symbol table. Called after all files are added and
  // before the writer writes results to a file.
  std::error_code addCombinedLTOObject();

  // The writer needs to handle DLL import libraries specially in
  // order to create the import descriptor table.
  std::vector<std::unique_ptr<ImportFile>> ImportFiles;

  // The writer needs to infer the machine type from the object files.
  std::vector<std::unique_ptr<ObjectFile>> ObjectFiles;

  // Creates an Undefined symbol for a given name.
  std::error_code addUndefined(StringRef Name);

  // Rename From -> To in the symbol table.
  std::error_code rename(StringRef From, StringRef To);

private:
  std::error_code addObject(ObjectFile *File);
  std::error_code addArchive(ArchiveFile *File);
  std::error_code addImport(ImportFile *File);
  std::error_code addBitcode(BitcodeFile *File);

  std::error_code resolve(SymbolBody *Body);
  std::error_code addMemberFile(Lazy *Body);
  std::error_code addSymbol(SymbolBody *Body);

  std::unordered_map<StringRef, Symbol *> Symtab;
  std::vector<std::unique_ptr<ArchiveFile>> ArchiveFiles;
  std::vector<std::unique_ptr<BitcodeFile>> BitcodeFiles;
  std::vector<std::unique_ptr<SymbolBody>> OwningSymbols;
  std::unique_ptr<MemoryBuffer> LTOObjectFile;
  llvm::BumpPtrAllocator Alloc;
};

} // namespace pecoff
} // namespace lld

#endif
