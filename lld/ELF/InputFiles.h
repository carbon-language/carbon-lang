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

#include "InputSection.h"
#include "Error.h"
#include "Symbols.h"

#include "lld/Core/LLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ELF.h"

namespace lld {
namespace elf2 {

using llvm::object::Archive;

class Lazy;
class SymbolBody;

// The root class of input files.
class InputFile {
public:
  enum Kind { ObjectKind, SharedKind, ArchiveKind };
  Kind kind() const { return FileKind; }
  virtual ~InputFile() {}

  // Reads a file (constructors don't do that).
  virtual void parse() = 0;

  StringRef getName() const { return MB.getBufferIdentifier(); }

protected:
  InputFile(Kind K, MemoryBufferRef M) : MB(M), FileKind(K) {}
  MemoryBufferRef MB;

private:
  const Kind FileKind;
};

enum ELFKind { ELF32LEKind, ELF32BEKind, ELF64LEKind, ELF64BEKind };

class ELFFileBase : public InputFile {
public:
  ELFFileBase(Kind K, ELFKind EKind, MemoryBufferRef M)
      : InputFile(K, M), EKind(EKind) {}
  static bool classof(const InputFile *F) {
    Kind K = F->kind();
    return K == ObjectKind || K == SharedKind;
  }

  bool isCompatibleWith(const ELFFileBase &Other) const;
  ELFKind getELFKind() const { return EKind; }

  uint16_t getEMachine() const;

protected:
  const ELFKind EKind;
};

// .o file.
class ObjectFileBase : public ELFFileBase {
public:
  ObjectFileBase(ELFKind EKind, MemoryBufferRef M)
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
  ELFData(MemoryBufferRef MB);
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym_Range Elf_Sym_Range;

  const llvm::object::ELFFile<ELFT> &getObj() const { return ELFObj; }
  llvm::object::ELFFile<ELFT> &getObj() { return ELFObj; }

  uint16_t getEMachine() const { return getObj().getHeader()->e_machine; }
  uint8_t getOSABI() const {
    return getObj().getHeader()->e_ident[llvm::ELF::EI_OSABI];
  }

  StringRef getStringTable() const { return StringTable; }

protected:
  llvm::object::ELFFile<ELFT> ELFObj;
  const Elf_Shdr *Symtab = nullptr;
  StringRef StringTable;
  Elf_Sym_Range getNonLocalSymbols();
  Elf_Sym_Range getSymbolsHelper(bool);
};

template <class ELFT>
class ObjectFile : public ObjectFileBase, public ELFData<ELFT> {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym_Range Elf_Sym_Range;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Word Elf_Word;

public:
  using ELFData<ELFT>::getEMachine;

  static bool classof(const InputFile *F) {
    return F->kind() == ObjectKind &&
           cast<ELFFileBase>(F)->getELFKind() == getStaticELFKind<ELFT>();
  }

  explicit ObjectFile(MemoryBufferRef M);
  void parse() override;

  ArrayRef<InputSection<ELFT> *> getSections() const { return Sections; }

  SymbolBody *getSymbolBody(uint32_t SymbolIndex) const {
    uint32_t FirstNonLocal = this->Symtab->sh_info;
    if (SymbolIndex < FirstNonLocal)
      return nullptr;
    return SymbolBodies[SymbolIndex - FirstNonLocal]->getReplacement();
  }

  Elf_Sym_Range getLocalSymbols();

  const Elf_Shdr *getSymbolTable() const { return this->Symtab; };
  ArrayRef<Elf_Word> getSymbolTableShndx() const { return SymtabSHNDX; };

private:
  void initializeSections();
  void initializeSymbols();

  SymbolBody *createSymbolBody(StringRef StringTable, const Elf_Sym *Sym);

  // List of all sections defined by this file.
  std::vector<InputSection<ELFT> *> Sections;

  ArrayRef<Elf_Word> SymtabSHNDX;
};

class ArchiveFile : public InputFile {
public:
  explicit ArchiveFile(MemoryBufferRef M) : InputFile(ArchiveKind, M) {}
  static bool classof(const InputFile *F) { return F->kind() == ArchiveKind; }
  void parse() override;

  // Returns a memory buffer for a given symbol. An empty memory buffer
  // is returned if we have already returned the same memory buffer.
  // (So that we don't instantiate same members more than once.)
  MemoryBufferRef getMember(const Archive::Symbol *Sym);

  llvm::MutableArrayRef<Lazy> getLazySymbols() { return LazySymbols; }

private:
  std::unique_ptr<Archive> File;
  std::vector<Lazy> LazySymbols;
  llvm::DenseSet<uint64_t> Seen;
};

// .so file.
class SharedFileBase : public ELFFileBase {
public:
  SharedFileBase(ELFKind EKind, MemoryBufferRef M)
      : ELFFileBase(SharedKind, EKind, M) {}
  static bool classof(const InputFile *F) { return F->kind() == SharedKind; }
};

template <class ELFT>
class SharedFile : public SharedFileBase, public ELFData<ELFT> {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym_Range Elf_Sym_Range;

  std::vector<SharedSymbol<ELFT>> SymbolBodies;

public:
  using ELFData<ELFT>::getEMachine;
  llvm::MutableArrayRef<SharedSymbol<ELFT>> getSharedSymbols() {
    return SymbolBodies;
  }

  static bool classof(const InputFile *F) {
    return F->kind() == SharedKind &&
           cast<ELFFileBase>(F)->getELFKind() == getStaticELFKind<ELFT>();
  }

  explicit SharedFile(MemoryBufferRef M);

  void parse() override;
};

template <template <class> class T>
std::unique_ptr<ELFFileBase> createELFFile(MemoryBufferRef MB) {
  using namespace llvm;

  std::pair<unsigned char, unsigned char> Type =
    object::getElfArchType(MB.getBuffer());
  if (Type.second != ELF::ELFDATA2LSB && Type.second != ELF::ELFDATA2MSB)
    error("Invalid data encoding: " + MB.getBufferIdentifier());

  if (Type.first == ELF::ELFCLASS32) {
    if (Type.second == ELF::ELFDATA2LSB)
      return make_unique<T<object::ELF32LE>>(MB);
    return make_unique<T<object::ELF32BE>>(MB);
  }
  if (Type.first == ELF::ELFCLASS64) {
    if (Type.second == ELF::ELFDATA2LSB)
      return make_unique<T<object::ELF64LE>>(MB);
    return make_unique<T<object::ELF64BE>>(MB);
  }
  error("Invalid file class: " + MB.getBufferIdentifier());
}

} // namespace elf2
} // namespace lld

#endif
