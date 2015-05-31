//===- InputFiles.h -------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_INPUT_FILES_H
#define LLD_COFF_INPUT_FILES_H

#include "Chunks.h"
#include "Memory.h"
#include "Symbols.h"
#include "lld/Core/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/COFF.h"
#include <memory>
#include <set>
#include <vector>

namespace lld {
namespace coff {

using llvm::object::Archive;
using llvm::object::COFFObjectFile;

// The root class of input files.
class InputFile {
public:
  enum Kind { ArchiveKind, ObjectKind, ImportKind };
  Kind kind() const { return FileKind; }
  virtual ~InputFile() {}

  // Returns the filename.
  virtual StringRef getName() = 0;

  // Returns symbols defined by this file.
  virtual std::vector<SymbolBody *> &getSymbols() = 0;

  // Reads a file (constructors don't do that). Returns an error if a
  // file is broken.
  virtual std::error_code parse() = 0;

  // Returns a short, human-friendly filename. If this is a member of
  // an archive file, a returned value includes parent's filename.
  // Used for logging or debugging.
  std::string getShortName();

  // Sets a parent filename if this file is created from an archive.
  void setParentName(StringRef N) { ParentName = N; }

protected:
  explicit InputFile(Kind K) : FileKind(K) {}

private:
  const Kind FileKind;
  StringRef ParentName;
};

// .lib or .a file.
class ArchiveFile : public InputFile {
public:
  explicit ArchiveFile(MemoryBufferRef M) : InputFile(ArchiveKind), MB(M) {}
  static bool classof(const InputFile *F) { return F->kind() == ArchiveKind; }
  std::error_code parse() override;
  StringRef getName() override { return Filename; }

  // Returns a memory buffer for a given symbol. An empty memory buffer
  // is returned if we have already returned the same memory buffer.
  // (So that we don't instantiate same members more than once.)
  ErrorOr<MemoryBufferRef> getMember(const Archive::Symbol *Sym);

  // NB: All symbols returned by ArchiveFiles are of Lazy type.
  std::vector<SymbolBody *> &getSymbols() override { return SymbolBodies; }

private:
  std::unique_ptr<Archive> File;
  std::string Filename;
  MemoryBufferRef MB;
  std::vector<SymbolBody *> SymbolBodies;
  std::set<const char *> Seen;
  llvm::MallocAllocator Alloc;
};

// .obj or .o file. This may be a member of an archive file.
class ObjectFile : public InputFile {
public:
  explicit ObjectFile(MemoryBufferRef M) : InputFile(ObjectKind), MB(M) {}
  static bool classof(const InputFile *F) { return F->kind() == ObjectKind; }
  std::error_code parse() override;
  StringRef getName() override { return MB.getBufferIdentifier(); }
  std::vector<Chunk *> &getChunks() { return Chunks; }
  std::vector<SymbolBody *> &getSymbols() override { return SymbolBodies; }

  // Returns a SymbolBody object for the SymbolIndex'th symbol in the
  // underlying object file.
  SymbolBody *getSymbolBody(uint32_t SymbolIndex);

  // Returns .drectve section contents if exist.
  StringRef getDirectives() { return Directives; }

  // Returns the underying COFF file.
  COFFObjectFile *getCOFFObj() { return COFFObj.get(); }

private:
  std::error_code initializeChunks();
  std::error_code initializeSymbols();

  SymbolBody *createSymbolBody(StringRef Name, COFFSymbolRef Sym,
                               const void *Aux, bool IsFirst);

  std::unique_ptr<COFFObjectFile> COFFObj;
  MemoryBufferRef MB;
  StringRef Directives;
  llvm::BumpPtrAllocator Alloc;

  // List of all chunks defined by this file. This includes both section
  // chunks and non-section chunks for common symbols.
  std::vector<Chunk *> Chunks;

  // This vector contains the same chunks as Chunks, but they are
  // indexed such that you can get a SectionChunk by section index.
  // Nonexistent section indices are filled with null pointers.
  // (Because section number is 1-based, the first slot is always a
  // null pointer.)
  std::vector<Chunk *> SparseChunks;

  // List of all symbols referenced or defined by this file.
  std::vector<SymbolBody *> SymbolBodies;

  // This vector contains the same symbols as SymbolBodies, but they
  // are indexed such that you can get a SymbolBody by symbol
  // index. Nonexistent indices (which are occupied by auxiliary
  // symbols in the real symbol table) are filled with null pointers.
  std::vector<SymbolBody *> SparseSymbolBodies;
};

// This type represents import library members that contain DLL names
// and symbols exported from the DLLs. See Microsoft PE/COFF spec. 7
// for details about the format.
class ImportFile : public InputFile {
public:
  explicit ImportFile(MemoryBufferRef M) : InputFile(ImportKind), MB(M) {}
  static bool classof(const InputFile *F) { return F->kind() == ImportKind; }
  StringRef getName() override { return MB.getBufferIdentifier(); }
  std::vector<SymbolBody *> &getSymbols() override { return SymbolBodies; }

private:
  std::error_code parse() override;

  MemoryBufferRef MB;
  std::vector<SymbolBody *> SymbolBodies;
  llvm::BumpPtrAllocator Alloc;
  StringAllocator StringAlloc;
};

} // namespace coff
} // namespace lld

#endif
