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

#include "Config.h"
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

class InputFile;
class Lazy;
class SymbolBody;

// The root class of input files.
class InputFile {
public:
  enum Kind { ObjectKind, SharedKind, ArchiveKind };
  Kind kind() const { return FileKind; }
  virtual ~InputFile() {}

  StringRef getName() const { return MB.getBufferIdentifier(); }

protected:
  InputFile(Kind K, MemoryBufferRef M) : MB(M), FileKind(K) {}
  MemoryBufferRef MB;

private:
  const Kind FileKind;
};

template <typename ELFT> class ELFFileBase : public InputFile {
public:
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym_Range Elf_Sym_Range;

  ELFFileBase(Kind K, ELFKind EKind, MemoryBufferRef M);
  static bool classof(const InputFile *F) {
    Kind K = F->kind();
    return K == ObjectKind || K == SharedKind;
  }

  ELFKind getELFKind() const { return EKind; }

  const llvm::object::ELFFile<ELFT> &getObj() const { return ELFObj; }
  llvm::object::ELFFile<ELFT> &getObj() { return ELFObj; }

  uint16_t getEMachine() const { return getObj().getHeader()->e_machine; }
  uint8_t getOSABI() const {
    return getObj().getHeader()->e_ident[llvm::ELF::EI_OSABI];
  }

  StringRef getStringTable() const { return StringTable; }

protected:
  const ELFKind EKind;
  llvm::object::ELFFile<ELFT> ELFObj;
  const Elf_Shdr *Symtab = nullptr;
  StringRef StringTable;
  void initStringTable();
  Elf_Sym_Range getNonLocalSymbols();
  Elf_Sym_Range getSymbolsHelper(bool);
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

// .o file.
template <class ELFT> class ObjectFile : public ELFFileBase<ELFT> {
  typedef ELFFileBase<ELFT> Base;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym_Range Elf_Sym_Range;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Word Elf_Word;

  typedef llvm::support::detail::packed_endian_specific_integral<
      uint32_t, ELFT::TargetEndianness, 2> GroupEntryType;
  StringRef getShtGroupSignature(const Elf_Shdr &Sec);
  ArrayRef<GroupEntryType> getShtGroupEntries(const Elf_Shdr &Sec);

public:
  static bool classof(const InputFile *F) {
    return F->kind() == Base::ObjectKind &&
           cast<ELFFileBase<ELFT>>(F)->getELFKind() == getStaticELFKind<ELFT>();
  }

  ArrayRef<SymbolBody *> getSymbols() { return this->SymbolBodies; }

  explicit ObjectFile(MemoryBufferRef M);
  void parse(llvm::DenseSet<StringRef> &Comdats);

  ArrayRef<InputSection<ELFT> *> getSections() const { return Sections; }

  SymbolBody *getSymbolBody(uint32_t SymbolIndex) const {
    uint32_t FirstNonLocal = this->Symtab->sh_info;
    if (SymbolIndex < FirstNonLocal)
      return nullptr;
    return this->SymbolBodies[SymbolIndex - FirstNonLocal];
  }

  Elf_Sym_Range getLocalSymbols();

  const Elf_Shdr *getSymbolTable() const { return this->Symtab; };
  ArrayRef<Elf_Word> getSymbolTableShndx() const { return SymtabSHNDX; };

private:
  void initializeSections(llvm::DenseSet<StringRef> &Comdats);
  void initializeSymbols();

  SymbolBody *createSymbolBody(StringRef StringTable, const Elf_Sym *Sym);

  // List of all sections defined by this file.
  std::vector<InputSection<ELFT> *> Sections;

  ArrayRef<Elf_Word> SymtabSHNDX;

  // List of all symbols referenced or defined by this file.
  std::vector<SymbolBody *> SymbolBodies;

  llvm::BumpPtrAllocator Alloc;
};

class ArchiveFile : public InputFile {
public:
  explicit ArchiveFile(MemoryBufferRef M) : InputFile(ArchiveKind, M) {}
  static bool classof(const InputFile *F) { return F->kind() == ArchiveKind; }
  void parse();

  // Returns a memory buffer for a given symbol. An empty memory buffer
  // is returned if we have already returned the same memory buffer.
  // (So that we don't instantiate same members more than once.)
  MemoryBufferRef getMember(const Archive::Symbol *Sym);

  llvm::MutableArrayRef<Lazy> getLazySymbols() { return LazySymbols; }
  std::vector<MemoryBufferRef> getMembers();

private:
  std::unique_ptr<Archive> File;
  std::vector<Lazy> LazySymbols;
  llvm::DenseSet<uint64_t> Seen;
};

// .so file.
template <class ELFT> class SharedFile : public ELFFileBase<ELFT> {
  typedef ELFFileBase<ELFT> Base;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym_Range Elf_Sym_Range;

  std::vector<SharedSymbol<ELFT>> SymbolBodies;
  StringRef SoName;

public:
  StringRef getSoName() const { return SoName; }
  llvm::MutableArrayRef<SharedSymbol<ELFT>> getSharedSymbols() {
    return SymbolBodies;
  }

  static bool classof(const InputFile *F) {
    return F->kind() == Base::SharedKind &&
           cast<ELFFileBase<ELFT>>(F)->getELFKind() == getStaticELFKind<ELFT>();
  }

  explicit SharedFile(MemoryBufferRef M);

  void parseSoName();
  void parse();

  // Used for --as-needed
  bool AsNeeded = false;
  bool IsUsed = false;
  bool isNeeded() const { return !AsNeeded || IsUsed; }
};

template <typename T>
std::unique_ptr<InputFile> createELFFileAux(MemoryBufferRef MB) {
  std::unique_ptr<T> Ret = llvm::make_unique<T>(MB);

  if (!Config->FirstElf)
    Config->FirstElf = Ret.get();

  if (Config->ElfKind == ELFNoneKind) {
    Config->ElfKind = Ret->getELFKind();
    Config->EMachine = Ret->getEMachine();
  }

  return std::move(Ret);
}

template <template <class> class T>
std::unique_ptr<InputFile> createELFFile(MemoryBufferRef MB) {
  using namespace llvm;

  std::pair<unsigned char, unsigned char> Type =
    object::getElfArchType(MB.getBuffer());
  if (Type.second != ELF::ELFDATA2LSB && Type.second != ELF::ELFDATA2MSB)
    error("Invalid data encoding: " + MB.getBufferIdentifier());

  if (Type.first == ELF::ELFCLASS32) {
    if (Type.second == ELF::ELFDATA2LSB)
      return createELFFileAux<T<object::ELF32LE>>(MB);
    return createELFFileAux<T<object::ELF32BE>>(MB);
  }
  if (Type.first == ELF::ELFCLASS64) {
    if (Type.second == ELF::ELFDATA2LSB)
      return createELFFileAux<T<object::ELF64LE>>(MB);
    return createELFFileAux<T<object::ELF64BE>>(MB);
  }
  error("Invalid file class: " + MB.getBufferIdentifier());
}

} // namespace elf2
} // namespace lld

#endif
