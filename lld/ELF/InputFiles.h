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
  enum Kind { ObjectKind, SharedKind };
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

class ELFFileBase : public InputFile {
public:
  explicit ELFFileBase(Kind K, ELFKind EKind, MemoryBufferRef M)
      : InputFile(K, M), EKind(EKind) {}
  static bool classof(const InputFile *F) {
    Kind K = F->kind();
    return K == ObjectKind || K == SharedKind;
  }

  bool isCompatibleWith(const ELFFileBase &Other) const;
  ELFKind getELFKind() const { return EKind; }

protected:
  const ELFKind EKind;
};

// .o file.
class ObjectFileBase : public ELFFileBase {
public:
  explicit ObjectFileBase(ELFKind EKind, MemoryBufferRef M)
      : ELFFileBase(ObjectKind, EKind, M) {}
  static bool classof(const InputFile *F) { return F->kind() == ObjectKind; }

  ArrayRef<SymbolBody *> getSymbols() { return SymbolBodies; }

protected:
  // List of all symbols referenced or defined by this file.
  std::vector<SymbolBody *> SymbolBodies;

  llvm::BumpPtrAllocator Alloc;
};

template <class ELFT> static ELFKind getStaticELFKind() {
  if (!ELFT::Is64Bits) {
    if (ELFT::TargetEndianness == llvm::support::little)
      return ELF32LEKind;
    return ELF32BEKind;
  }
  if (ELFT::TargetEndianness == llvm::support::little)
    return ELF64LEKind;
  return ELF64BEKind;
}

template <class ELFT> class ELFData {
public:
  llvm::object::ELFFile<ELFT> *getObj() const { return ELFObj.get(); }

  uint16_t getEMachine() const { return getObj()->getHeader()->e_machine; }

protected:
  std::unique_ptr<llvm::object::ELFFile<ELFT>> ELFObj;

  void openELF(MemoryBufferRef MB);
};

template <class ELFT>
class ObjectFile : public ObjectFileBase, public ELFData<ELFT> {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym_Range Elf_Sym_Range;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Word Elf_Word;

public:

  static bool classof(const InputFile *F) {
    return F->kind() == ObjectKind &&
           cast<ELFFileBase>(F)->getELFKind() == getStaticELFKind<ELFT>();
  }

  explicit ObjectFile(MemoryBufferRef M)
      : ObjectFileBase(getStaticELFKind<ELFT>(), M) {}
  void parse() override;

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

  // List of all chunks defined by this file.
  std::vector<SectionChunk<ELFT> *> Chunks;

  const Elf_Shdr *Symtab = nullptr;
  ArrayRef<Elf_Word> SymtabSHNDX;
};

// .so file.
class SharedFileBase : public ELFFileBase {
public:
  explicit SharedFileBase(ELFKind EKind, MemoryBufferRef M)
      : ELFFileBase(SharedKind, EKind, M) {}
  static bool classof(const InputFile *F) { return F->kind() == SharedKind; }
};

template <class ELFT>
class SharedFile : public SharedFileBase, public ELFData<ELFT> {
public:
  static bool classof(const InputFile *F) {
    return F->kind() == SharedKind &&
           cast<ELFFileBase>(F)->getELFKind() == getStaticELFKind<ELFT>();
  }

  explicit SharedFile(MemoryBufferRef M)
      : SharedFileBase(getStaticELFKind<ELFT>(), M) {}

  void parse() override;
};

} // namespace elf2
} // namespace lld

#endif
