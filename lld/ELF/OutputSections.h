//===- OutputSections.h -----------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_OUTPUT_SECTIONS_H
#define LLD_ELF_OUTPUT_SECTIONS_H

#include "lld/Core/LLVM.h"

#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Object/ELF.h"

#include "Config.h"

#include <type_traits>

namespace lld {
namespace elf2 {

class SymbolBody;
template <class ELFT> class SymbolTable;
template <class ELFT> class SymbolTableSection;
template <class ELFT> class StringTableSection;
template <class ELFT> class InputSection;
template <class ELFT> class OutputSection;
template <class ELFT> class ObjectFile;
template <class ELFT> class DefinedRegular;
template <class ELFT> class ELFSymbolBody;

template <class ELFT>
typename llvm::object::ELFFile<ELFT>::uintX_t getSymVA(const SymbolBody &S);

template <class ELFT>
typename llvm::object::ELFFile<ELFT>::uintX_t
getLocalRelTarget(const ObjectFile<ELFT> &File,
                  const typename llvm::object::ELFFile<ELFT>::Elf_Rel &Sym);
bool canBePreempted(const SymbolBody *Body, bool NeedsGot);
template <class ELFT> bool includeInSymtab(const SymbolBody &B);

bool includeInDynamicSymtab(const SymbolBody &B);

template <class ELFT>
bool shouldKeepInSymtab(
    const ObjectFile<ELFT> &File, StringRef Name,
    const typename llvm::object::ELFFile<ELFT>::Elf_Sym &Sym);

// This represents a section in an output file.
// Different sub classes represent different types of sections. Some contain
// input sections, others are created by the linker.
// The writer creates multiple OutputSections and assign them unique,
// non-overlapping file offsets and VAs.
template <class ELFT> class OutputSectionBase {
public:
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;

  OutputSectionBase(StringRef Name, uint32_t sh_type, uintX_t sh_flags);
  void setVA(uintX_t VA) { Header.sh_addr = VA; }
  uintX_t getVA() const { return Header.sh_addr; }
  void setFileOffset(uintX_t Off) { Header.sh_offset = Off; }
  void writeHeaderTo(Elf_Shdr *SHdr);
  StringRef getName() { return Name; }
  void setNameOffset(uintX_t Offset) { Header.sh_name = Offset; }

  unsigned SectionIndex;

  // Returns the size of the section in the output file.
  uintX_t getSize() const { return Header.sh_size; }
  void setSize(uintX_t Val) { Header.sh_size = Val; }
  uintX_t getFlags() { return Header.sh_flags; }
  uintX_t getFileOff() { return Header.sh_offset; }
  uintX_t getAlign() {
    // The ELF spec states that a value of 0 means the section has no alignment
    // constraits.
    return std::max<uintX_t>(Header.sh_addralign, 1);
  }
  uint32_t getType() { return Header.sh_type; }

  virtual void finalize() {}
  virtual void writeTo(uint8_t *Buf) = 0;

protected:
  StringRef Name;
  Elf_Shdr Header;
  ~OutputSectionBase() = default;
};

template <class ELFT> class GotSection final : public OutputSectionBase<ELFT> {
  typedef OutputSectionBase<ELFT> Base;
  typedef typename Base::uintX_t uintX_t;

public:
  GotSection();
  void finalize() override {
    this->Header.sh_size = Entries.size() * sizeof(uintX_t);
  }
  void writeTo(uint8_t *Buf) override;
  void addEntry(SymbolBody *Sym);
  bool empty() const { return Entries.empty(); }
  uintX_t getEntryAddr(const SymbolBody &B) const;

private:
  std::vector<const SymbolBody *> Entries;
};

template <class ELFT> class PltSection final : public OutputSectionBase<ELFT> {
  typedef OutputSectionBase<ELFT> Base;
  typedef typename Base::uintX_t uintX_t;

public:
  PltSection();
  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  void addEntry(SymbolBody *Sym);
  bool empty() const { return Entries.empty(); }
  uintX_t getEntryAddr(const SymbolBody &B) const;

private:
  std::vector<const SymbolBody *> Entries;
};

template <class ELFT> struct DynamicReloc {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rel Elf_Rel;
  const InputSection<ELFT> &C;
  const Elf_Rel &RI;
};

template <class ELFT>
class SymbolTableSection final : public OutputSectionBase<ELFT> {
public:
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym_Range Elf_Sym_Range;
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;
  SymbolTableSection(SymbolTable<ELFT> &Table,
                     StringTableSection<ELFT> &StrTabSec);

  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  void addSymbol(StringRef Name, bool isLocal = false);
  StringTableSection<ELFT> &getStrTabSec() const { return StrTabSec; }
  unsigned getNumSymbols() const { return NumVisible + 1; }

private:
  void writeLocalSymbols(uint8_t *&Buf);
  void writeGlobalSymbols(uint8_t *&Buf);

  SymbolTable<ELFT> &Table;
  StringTableSection<ELFT> &StrTabSec;
  unsigned NumVisible = 0;
  unsigned NumLocals = 0;
};

template <class ELFT>
class RelocationSection final : public OutputSectionBase<ELFT> {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rel Elf_Rel;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rela Elf_Rela;
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;

public:
  RelocationSection(bool IsRela);
  void addReloc(const DynamicReloc<ELFT> &Reloc) { Relocs.push_back(Reloc); }
  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  bool hasRelocs() const { return !Relocs.empty(); }
  bool isRela() const { return IsRela; }

private:
  std::vector<DynamicReloc<ELFT>> Relocs;
  const bool IsRela;
};

template <class ELFT>
class OutputSection final : public OutputSectionBase<ELFT> {
public:
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rel Elf_Rel;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rela Elf_Rela;
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;
  OutputSection(StringRef Name, uint32_t sh_type, uintX_t sh_flags);
  void addSection(InputSection<ELFT> *C);
  void writeTo(uint8_t *Buf) override;

private:
  std::vector<InputSection<ELFT> *> Sections;
};

template <class ELFT>
class InterpSection final : public OutputSectionBase<ELFT> {
public:
  InterpSection();
  void writeTo(uint8_t *Buf);
};

template <class ELFT>
class StringTableSection final : public OutputSectionBase<ELFT> {
public:
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;
  StringTableSection(bool Dynamic);
  void add(StringRef S) { StrTabBuilder.add(S); }
  size_t getFileOff(StringRef S) const { return StrTabBuilder.getOffset(S); }
  StringRef data() const { return StrTabBuilder.data(); }
  void writeTo(uint8_t *Buf) override;

  void finalize() override {
    StrTabBuilder.finalize(llvm::StringTableBuilder::ELF);
    this->Header.sh_size = StrTabBuilder.data().size();
  }

  bool isDynamic() const { return Dynamic; }

private:
  const bool Dynamic;
  llvm::StringTableBuilder StrTabBuilder;
};

template <class ELFT>
class HashTableSection final : public OutputSectionBase<ELFT> {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Word Elf_Word;

public:
  HashTableSection();
  void addSymbol(SymbolBody *S);
  void finalize() override;
  void writeTo(uint8_t *Buf) override;

private:
  std::vector<uint32_t> Hashes;
};

template <class ELFT>
class DynamicSection final : public OutputSectionBase<ELFT> {
  typedef OutputSectionBase<ELFT> Base;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Dyn Elf_Dyn;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rel Elf_Rel;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rela Elf_Rela;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;

public:
  DynamicSection(SymbolTable<ELFT> &SymTab);
  void finalize() override;
  void writeTo(uint8_t *Buf) override;

  OutputSectionBase<ELFT> *PreInitArraySec = nullptr;
  OutputSectionBase<ELFT> *InitArraySec = nullptr;
  OutputSectionBase<ELFT> *FiniArraySec = nullptr;

private:
  SymbolTable<ELFT> &SymTab;
  const ELFSymbolBody<ELFT> *InitSym = nullptr;
  const ELFSymbolBody<ELFT> *FiniSym = nullptr;
};

// All output sections that are hadnled by the linker specially are
// globally accessible. Writer initializes them, so don't use them
// until Writer is initialized.
template <class ELFT> struct Out {
  static DynamicSection<ELFT> *Dynamic;
  static GotSection<ELFT> *Got;
  static HashTableSection<ELFT> *HashTab;
  static InterpSection<ELFT> *Interp;
  static OutputSection<ELFT> *Bss;
  static OutputSectionBase<ELFT> *Opd;
  static uint8_t *OpdBuf;
  static PltSection<ELFT> *Plt;
  static RelocationSection<ELFT> *RelaDyn;
  static StringTableSection<ELFT> *DynStrTab;
  static StringTableSection<ELFT> *StrTab;
  static SymbolTableSection<ELFT> *DynSymTab;
  static SymbolTableSection<ELFT> *SymTab;
};

template <class ELFT> DynamicSection<ELFT> *Out<ELFT>::Dynamic;
template <class ELFT> GotSection<ELFT> *Out<ELFT>::Got;
template <class ELFT> HashTableSection<ELFT> *Out<ELFT>::HashTab;
template <class ELFT> InterpSection<ELFT> *Out<ELFT>::Interp;
template <class ELFT> OutputSection<ELFT> *Out<ELFT>::Bss;
template <class ELFT> OutputSectionBase<ELFT> *Out<ELFT>::Opd;
template <class ELFT> uint8_t *Out<ELFT>::OpdBuf;
template <class ELFT> PltSection<ELFT> *Out<ELFT>::Plt;
template <class ELFT> RelocationSection<ELFT> *Out<ELFT>::RelaDyn;
template <class ELFT> StringTableSection<ELFT> *Out<ELFT>::DynStrTab;
template <class ELFT> StringTableSection<ELFT> *Out<ELFT>::StrTab;
template <class ELFT> SymbolTableSection<ELFT> *Out<ELFT>::DynSymTab;
template <class ELFT> SymbolTableSection<ELFT> *Out<ELFT>::SymTab;
}
}
#endif
