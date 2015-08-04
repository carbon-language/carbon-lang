//===- InputFiles.h -------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_INPUT_FILES_H
#define LLD_ELF_INPUT_FILES_H

#include "lld/Core/LLVM.h"
#include "llvm/Object/ELF.h"

namespace lld {
namespace elf2 {
class SymbolBody;
class Chunk;

// The root class of input files.
class InputFile {
public:
  enum Kind { ObjectKind };
  Kind kind() const { return FileKind; }
  virtual ~InputFile() {}

  // Returns symbols defined by this file.
  virtual ArrayRef<SymbolBody *> getSymbols() = 0;

  // Reads a file (constructors don't do that).
  virtual void parse() = 0;

protected:
  explicit InputFile(Kind K, MemoryBufferRef M) : MB(M), FileKind(K) {}
  MemoryBufferRef MB;

private:
  const Kind FileKind;
};

// .o file.
class ObjectFileBase : public InputFile {
public:
  explicit ObjectFileBase(MemoryBufferRef M) : InputFile(ObjectKind, M) {}
  static bool classof(const InputFile *F) { return F->kind() == ObjectKind; }

  ArrayRef<Chunk *> getChunks() { return Chunks; }
  ArrayRef<SymbolBody *> getSymbols() override { return SymbolBodies; }

protected:
  // List of all chunks defined by this file. This includes both section
  // chunks and non-section chunks for common symbols.
  std::vector<Chunk *> Chunks;

  // List of all symbols referenced or defined by this file.
  std::vector<SymbolBody *> SymbolBodies;

  llvm::BumpPtrAllocator Alloc;
};

template <class ELFT> class ObjectFile : public ObjectFileBase {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym_Range Elf_Sym_Range;

public:
  explicit ObjectFile(MemoryBufferRef M) : ObjectFileBase(M) {}
  void parse() override;

  // Returns the underying ELF file.
  llvm::object::ELFFile<ELFT> *getObj() { return ELFObj.get(); }

private:
  void initializeChunks();
  void initializeSymbols();

  SymbolBody *createSymbolBody(StringRef StringTable, const Elf_Sym *Sym);

  std::unique_ptr<llvm::object::ELFFile<ELFT>> ELFObj;
};

} // namespace elf2
} // namespace lld

#endif
