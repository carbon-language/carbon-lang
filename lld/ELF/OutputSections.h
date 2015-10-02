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
class SymbolTable;
template <class ELFT> class SymbolTableSection;
template <bool Is64Bits> class StringTableSection;
template <class ELFT> class InputSection;
template <class ELFT> class OutputSection;
template <class ELFT> class ObjectFile;
template <class ELFT> class DefinedRegular;
template <class ELFT> class ELFSymbolBody;

template <class ELFT>
typename llvm::object::ELFFile<ELFT>::uintX_t
getSymVA(const ELFSymbolBody<ELFT> &S, const OutputSection<ELFT> &BssSec);

template <class ELFT>
typename llvm::object::ELFFile<ELFT>::uintX_t
getLocalSymVA(const typename llvm::object::ELFFile<ELFT>::Elf_Sym *Sym,
              const ObjectFile<ELFT> &File);

bool includeInSymtab(const SymbolBody &B);
bool includeInDynamicSymtab(const SymbolBody &B);
bool shouldKeepInSymtab(StringRef SymName);

// This represents a section in an output file.
// Different sub classes represent different types of sections. Some contain
// input sections, others are created by the linker.
// The writer creates multiple OutputSections and assign them unique,
// non-overlapping file offsets and VAs.
template <bool Is64Bits> class OutputSectionBase {
public:
  typedef typename std::conditional<Is64Bits, uint64_t, uint32_t>::type uintX_t;
  typedef typename std::conditional<Is64Bits, llvm::ELF::Elf64_Shdr,
                                    llvm::ELF::Elf32_Shdr>::type HeaderT;

  OutputSectionBase(StringRef Name, uint32_t sh_type, uintX_t sh_flags);
  void setVA(uintX_t VA) { Header.sh_addr = VA; }
  uintX_t getVA() const { return Header.sh_addr; }
  void setFileOffset(uintX_t Off) { Header.sh_offset = Off; }
  template <llvm::object::endianness E>
  void writeHeaderTo(typename llvm::object::ELFFile<
                     llvm::object::ELFType<E, Is64Bits>>::Elf_Shdr *SHdr);
  StringRef getName() { return Name; }
  void setNameOffset(uintX_t Offset) { Header.sh_name = Offset; }

  unsigned getSectionIndex() const { return SectionIndex; }
  void setSectionIndex(unsigned I) { SectionIndex = I; }

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

  static unsigned getAddrSize() { return Is64Bits ? 8 : 4; }

  virtual void finalize() {}
  virtual void writeTo(uint8_t *Buf) = 0;

protected:
  StringRef Name;
  HeaderT Header;
  unsigned SectionIndex;
  ~OutputSectionBase() = default;
};

template <class ELFT>
class GotSection final : public OutputSectionBase<ELFT::Is64Bits> {
  typedef OutputSectionBase<ELFT::Is64Bits> Base;
  typedef typename Base::uintX_t uintX_t;

public:
  GotSection();
  void finalize() override {
    this->Header.sh_size = Entries.size() * this->getAddrSize();
  }
  void writeTo(uint8_t *Buf) override {}
  void addEntry(SymbolBody *Sym);
  bool empty() const { return Entries.empty(); }
  uintX_t getEntryAddr(const SymbolBody &B) const;

private:
  std::vector<const SymbolBody *> Entries;
};

template <class ELFT>
class PltSection final : public OutputSectionBase<ELFT::Is64Bits> {
  typedef OutputSectionBase<ELFT::Is64Bits> Base;
  typedef typename Base::uintX_t uintX_t;

public:
  PltSection(const GotSection<ELFT> &GotSec);
  void finalize() override {
    this->Header.sh_size = Entries.size() * EntrySize;
  }
  void writeTo(uint8_t *Buf) override;
  void addEntry(SymbolBody *Sym);
  bool empty() const { return Entries.empty(); }
  uintX_t getEntryAddr(const SymbolBody &B) const;
  static const unsigned EntrySize = 8;

private:
  std::vector<const SymbolBody *> Entries;
  const GotSection<ELFT> &GotSec;
};

template <class ELFT> struct DynamicReloc {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rel Elf_Rel;
  const InputSection<ELFT> &C;
  const Elf_Rel &RI;
};

template <class ELFT>
class SymbolTableSection final : public OutputSectionBase<ELFT::Is64Bits> {
public:
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym_Range Elf_Sym_Range;
  typedef typename OutputSectionBase<ELFT::Is64Bits>::uintX_t uintX_t;
  SymbolTableSection(SymbolTable &Table,
                     StringTableSection<ELFT::Is64Bits> &StrTabSec,
                     const OutputSection<ELFT> &BssSec);

  void finalize() override {
    this->Header.sh_size = getNumSymbols() * sizeof(Elf_Sym);
    this->Header.sh_link = StrTabSec.getSectionIndex();
    this->Header.sh_info = NumLocals + 1;
  }

  void writeTo(uint8_t *Buf) override;

  SymbolTable &getSymTable() const { return Table; }

  void addSymbol(StringRef Name, bool isLocal = false) {
    StrTabSec.add(Name);
    ++NumVisible;
    if (isLocal)
      ++NumLocals;
  }

  StringTableSection<ELFT::Is64Bits> &getStrTabSec() const { return StrTabSec; }
  unsigned getNumSymbols() const { return NumVisible + 1; }

private:
  void writeLocalSymbols(uint8_t *&Buf);
  void writeGlobalSymbols(uint8_t *&Buf);

  SymbolTable &Table;
  StringTableSection<ELFT::Is64Bits> &StrTabSec;
  unsigned NumVisible = 0;
  unsigned NumLocals = 0;
  const OutputSection<ELFT> &BssSec;
};

template <class ELFT>
class RelocationSection final : public OutputSectionBase<ELFT::Is64Bits> {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rel Elf_Rel;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rela Elf_Rela;

public:
  RelocationSection(SymbolTableSection<ELFT> &DynSymSec,
                    const GotSection<ELFT> &GotSec, bool IsRela);
  void addReloc(const DynamicReloc<ELFT> &Reloc) { Relocs.push_back(Reloc); }
  void finalize() override;
  void writeTo(uint8_t *Buf) override;
  bool hasRelocs() const { return !Relocs.empty(); }
  bool isRela() const { return IsRela; }

private:
  std::vector<DynamicReloc<ELFT>> Relocs;
  SymbolTableSection<ELFT> &DynSymSec;
  const GotSection<ELFT> &GotSec;
  const bool IsRela;
};

template <class ELFT>
class OutputSection final : public OutputSectionBase<ELFT::Is64Bits> {
public:
  typedef typename OutputSectionBase<ELFT::Is64Bits>::uintX_t uintX_t;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rel Elf_Rel;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rela Elf_Rela;
  OutputSection(const PltSection<ELFT> &PltSec, const GotSection<ELFT> &GotSec,
                const OutputSection<ELFT> &BssSec, StringRef Name,
                uint32_t sh_type, uintX_t sh_flags);
  void addSection(InputSection<ELFT> *C);
  void writeTo(uint8_t *Buf) override;

private:
  std::vector<InputSection<ELFT> *> Sections;
  const PltSection<ELFT> &PltSec;
  const GotSection<ELFT> &GotSec;
  const OutputSection<ELFT> &BssSec;
};

template <bool Is64Bits>
class InterpSection final : public OutputSectionBase<Is64Bits> {
public:
  InterpSection();

  void writeTo(uint8_t *Buf);
};

template <bool Is64Bits>
class StringTableSection final : public OutputSectionBase<Is64Bits> {
public:
  typedef typename OutputSectionBase<Is64Bits>::uintX_t uintX_t;
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
class HashTableSection final : public OutputSectionBase<ELFT::Is64Bits> {
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Word Elf_Word;

public:
  HashTableSection(SymbolTableSection<ELFT> &DynSymSec);
  void addSymbol(SymbolBody *S);

  void finalize() override {
    this->Header.sh_link = DynSymSec.getSectionIndex();

    assert(DynSymSec.getNumSymbols() == Hashes.size() + 1);
    unsigned NumEntries = 2;                 // nbucket and nchain.
    NumEntries += DynSymSec.getNumSymbols(); // The chain entries.

    // Create as many buckets as there are symbols.
    // FIXME: This is simplistic. We can try to optimize it, but implementing
    // support for SHT_GNU_HASH is probably even more profitable.
    NumEntries += DynSymSec.getNumSymbols();
    this->Header.sh_size = NumEntries * sizeof(Elf_Word);
  }

  void writeTo(uint8_t *Buf) override {
    unsigned NumSymbols = DynSymSec.getNumSymbols();
    auto *P = reinterpret_cast<Elf_Word *>(Buf);
    *P++ = NumSymbols; // nbucket
    *P++ = NumSymbols; // nchain

    Elf_Word *Buckets = P;
    Elf_Word *Chains = P + NumSymbols;

    for (unsigned I = 1; I < NumSymbols; ++I) {
      uint32_t Hash = Hashes[I - 1] % NumSymbols;
      Chains[I] = Buckets[Hash];
      Buckets[Hash] = I;
    }
  }

  SymbolTableSection<ELFT> &getDynSymSec() { return DynSymSec; }

private:
  uint32_t hash(StringRef Name) {
    uint32_t H = 0;
    for (char C : Name) {
      H = (H << 4) + C;
      uint32_t G = H & 0xf0000000;
      if (G)
        H ^= G >> 24;
      H &= ~G;
    }
    return H;
  }
  SymbolTableSection<ELFT> &DynSymSec;
  std::vector<uint32_t> Hashes;
};

template <class ELFT>
class DynamicSection final : public OutputSectionBase<ELFT::Is64Bits> {
  typedef OutputSectionBase<ELFT::Is64Bits> Base;
  typedef typename Base::HeaderT HeaderT;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rel Elf_Rel;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rela Elf_Rela;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Dyn Elf_Dyn;

public:
  DynamicSection(SymbolTable &SymTab, HashTableSection<ELFT> &HashSec,
                 RelocationSection<ELFT> &RelaDynSec);
  void finalize() override;
  void writeTo(uint8_t *Buf) override;

  OutputSection<ELFT> *PreInitArraySec = nullptr;
  OutputSection<ELFT> *InitArraySec = nullptr;
  OutputSection<ELFT> *FiniArraySec = nullptr;

private:
  HashTableSection<ELFT> &HashSec;
  SymbolTableSection<ELFT> &DynSymSec;
  StringTableSection<ELFT::Is64Bits> &DynStrSec;
  RelocationSection<ELFT> &RelaDynSec;
  SymbolTable &SymTab;
};
}
}
#endif
