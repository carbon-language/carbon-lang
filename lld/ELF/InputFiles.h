//===- InputFiles.h ---------------------------------------------*- C++ -*-===//
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
#include "Symbols.h"

#include "lld/Core/LLVM.h"
#include "llvm/Object/ELF.h"

namespace lld {
namespace elf2 {
class SymbolBody;

// The root class of input files.
class InputFile {
public:
  enum Kind { ObjectKind };
  Kind kind() const { return FileKind; }
  virtual ~InputFile() {}

  // Reads a file (constructors don't do that).
  virtual void parse() = 0;

  StringRef getName() const { return MB.getBufferIdentifier(); }

protected:
  explicit InputFile(Kind K, MemoryBufferRef M) : MB(M), FileKind(K) {}
  MemoryBufferRef MB;

private:
  const Kind FileKind;
};

enum ELFKind { ELF32LEKind, ELF32BEKind, ELF64LEKind, ELF64BEKind };

// .o file.
class ObjectFileBase : public InputFile {
public:
  explicit ObjectFileBase(ELFKind EKind, MemoryBufferRef M)
      : InputFile(ObjectKind, M), EKind(EKind) {}
  static bool classof(const InputFile *F) { return F->kind() == ObjectKind; }

  ArrayRef<SymbolBody *> getSymbols() { return SymbolBodies; }

  virtual bool isCompatibleWith(const ObjectFileBase &Other) const = 0;

  ELFKind getELFKind() const { return EKind; }

protected:
  // List of all symbols referenced or defined by this file.
  std::vector<SymbolBody *> SymbolBodies;

  llvm::BumpPtrAllocator Alloc;
  const ELFKind EKind;
};

template <class ELFT> class ObjectFile : public ObjectFileBase {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym_Range Elf_Sym_Range;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Word Elf_Word;

public:
  bool isCompatibleWith(const ObjectFileBase &Other) const override;

  static ELFKind getStaticELFKind() {
    if (!ELFT::Is64Bits) {
      if (ELFT::TargetEndianness == llvm::support::little)
        return ELF32LEKind;
      return ELF32BEKind;
    }
    if (ELFT::TargetEndianness == llvm::support::little)
      return ELF64LEKind;
    return ELF64BEKind;
  }

  static bool classof(const InputFile *F) {
    return F->kind() == ObjectKind &&
           cast<ObjectFileBase>(F)->getELFKind() == getStaticELFKind();
  }

  explicit ObjectFile(MemoryBufferRef M)
      : ObjectFileBase(getStaticELFKind(), M) {}
  void parse() override;

  // Returns the underying ELF file.
  llvm::object::ELFFile<ELFT> *getObj() const { return ELFObj.get(); }

  ArrayRef<SectionChunk<ELFT> *> getChunks() { return Chunks; }

  SymbolBody *getSymbolBody(uint32_t SymbolIndex) {
    uint32_t FirstNonLocal = Symtab->sh_info;
    if (SymbolIndex < FirstNonLocal)
      return nullptr;
    return SymbolBodies[SymbolIndex - FirstNonLocal]->getReplacement();
  }

private:
  void initializeChunks();
  void initializeSymbols();

  SymbolBody *createSymbolBody(StringRef StringTable, const Elf_Sym *Sym);

  std::unique_ptr<llvm::object::ELFFile<ELFT>> ELFObj;

  // List of all chunks defined by this file.
  std::vector<SectionChunk<ELFT> *> Chunks;

  const Elf_Shdr *Symtab = nullptr;
  ArrayRef<Elf_Word> SymtabSHNDX;
};

} // namespace elf2
} // namespace lld

#endif
