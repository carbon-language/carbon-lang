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

#include "Chunks.h"
#include "lld/Core/LLVM.h"
#include "llvm/Object/ELF.h"

namespace lld {
namespace elf2 {
class SymbolBody;

// The root class of input files.
class InputFile {
public:
  enum Kind { Object32LEKind, Object32BEKind, Object64LEKind, Object64BEKind };
  Kind kind() const { return FileKind; }
  virtual ~InputFile() {}

  // Returns symbols defined by this file.
  virtual ArrayRef<SymbolBody *> getSymbols() = 0;

  // Reads a file (constructors don't do that).
  virtual void parse() = 0;

  StringRef getName() const { return MB.getBufferIdentifier(); }

protected:
  explicit InputFile(Kind K, MemoryBufferRef M) : MB(M), FileKind(K) {}
  MemoryBufferRef MB;

private:
  const Kind FileKind;
};

// .o file.
class ObjectFileBase : public InputFile {
public:
  explicit ObjectFileBase(Kind K, MemoryBufferRef M) : InputFile(K, M) {}
  static bool classof(const InputFile *F) {
    Kind K = F->kind();
    return K >= Object32LEKind && K <= Object64BEKind;
  }

  ArrayRef<SymbolBody *> getSymbols() override { return SymbolBodies; }

  virtual bool isCompatibleWith(const ObjectFileBase &Other) const = 0;

protected:
  // List of all symbols referenced or defined by this file.
  std::vector<SymbolBody *> SymbolBodies;

  llvm::BumpPtrAllocator Alloc;
};

template <class ELFT> class ObjectFile : public ObjectFileBase {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym_Range Elf_Sym_Range;

public:
  bool isCompatibleWith(const ObjectFileBase &Other) const override;

  static Kind getKind() {
    if (!ELFT::Is64Bits) {
      if (ELFT::TargetEndianness == llvm::support::little)
        return Object32LEKind;
      return Object32BEKind;
    }
    if (ELFT::TargetEndianness == llvm::support::little)
      return Object64LEKind;
    return Object64BEKind;
  }

  static bool classof(const InputFile *F) { return F->kind() == getKind(); }

  explicit ObjectFile(MemoryBufferRef M) : ObjectFileBase(getKind(), M) {}
  void parse() override;

  // Returns the underying ELF file.
  llvm::object::ELFFile<ELFT> *getObj() const { return ELFObj.get(); }

  ArrayRef<SectionChunk<ELFT> *> getChunks() { return Chunks; }

private:
  void initializeChunks();
  void initializeSymbols();

  SymbolBody *createSymbolBody(StringRef StringTable, const Elf_Sym *Sym);

  std::unique_ptr<llvm::object::ELFFile<ELFT>> ELFObj;

  // List of all chunks defined by this file.
  std::vector<SectionChunk<ELFT> *> Chunks;

  const Elf_Shdr *Symtab = nullptr;
};

} // namespace elf2
} // namespace lld

#endif
