//===- lib/ReaderWriter/ELF/HexagonELFFile.h ------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_HEXAGON_ELF_FILE_H
#define LLD_READER_WRITER_ELF_HEXAGON_ELF_FILE_H

#include "ELFReader.h"
#include "HexagonLinkingContext.h"

namespace lld {
namespace elf {

template <class ELFT> class HexagonELFFile;

template <class ELFT>
class HexagonELFDefinedAtom : public ELFDefinedAtom<ELFT> {
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;
  typedef llvm::object::Elf_Shdr_Impl<ELFT> Elf_Shdr;

public:
  template<typename... T>
  HexagonELFDefinedAtom(T&&... args)
      : ELFDefinedAtom<ELFT>(std::forward<T>(args)...) {}

  virtual DefinedAtom::ContentType contentType() const {
    if (this->_contentType != DefinedAtom::typeUnknown)
      return this->_contentType;
    else if (this->_section->sh_flags & llvm::ELF::SHF_HEX_GPREL) {
      if (this->_section->sh_type == llvm::ELF::SHT_NOBITS)
        return (this->_contentType = DefinedAtom::typeZeroFillFast);
      else
        return (this->_contentType = DefinedAtom::typeDataFast);
    }
    return ELFDefinedAtom<ELFT>::contentType();
  }

  virtual DefinedAtom::ContentPermissions permissions() const {
    if (this->_section->sh_flags & llvm::ELF::SHF_HEX_GPREL)
      return DefinedAtom::permRW_;
    return ELFDefinedAtom<ELFT>::permissions();
  }
};

template <class ELFT> class HexagonELFCommonAtom : public ELFCommonAtom<ELFT> {
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;
  typedef llvm::object::Elf_Shdr_Impl<ELFT> Elf_Shdr;

public:
  HexagonELFCommonAtom(const HexagonELFFile<ELFT> &file, StringRef symbolName,
                       const Elf_Sym *symbol)
      : ELFCommonAtom<ELFT>(file, symbolName, symbol) {}

  virtual bool isSmallCommonSymbol() const {
    switch (this->_symbol->st_shndx) {
    // Common symbols
    case llvm::ELF::SHN_HEXAGON_SCOMMON:
    case llvm::ELF::SHN_HEXAGON_SCOMMON_1:
    case llvm::ELF::SHN_HEXAGON_SCOMMON_2:
    case llvm::ELF::SHN_HEXAGON_SCOMMON_4:
    case llvm::ELF::SHN_HEXAGON_SCOMMON_8:
      return true;
    default:
      break;
    }
    return false;
  }

  virtual uint64_t size() const {
    if (isSmallCommonSymbol())
      return this->_symbol->st_size;
    return ELFCommonAtom<ELFT>::size();
  }

  virtual DefinedAtom::Merge merge() const {
    if (this->_symbol->getBinding() == llvm::ELF::STB_WEAK)
      return DefinedAtom::mergeAsWeak;
    if (isSmallCommonSymbol())
      return DefinedAtom::mergeAsTentative;
    return ELFCommonAtom<ELFT>::merge();
  }

  virtual DefinedAtom::ContentType contentType() const {
    if (isSmallCommonSymbol())
      return DefinedAtom::typeZeroFillFast;
    return ELFCommonAtom<ELFT>::contentType();
  }

  virtual DefinedAtom::Alignment alignment() const {
    if (isSmallCommonSymbol())
      return DefinedAtom::Alignment(this->_symbol->st_value);
    return 1;
  }

  virtual DefinedAtom::ContentPermissions permissions() const {
    if (isSmallCommonSymbol())
      return DefinedAtom::permRW_;
    return ELFCommonAtom<ELFT>::permissions();
  }
};

template <class ELFT> class HexagonELFFile : public ELFFile<ELFT> {
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;
  typedef llvm::object::Elf_Shdr_Impl<ELFT> Elf_Shdr;

public:
  HexagonELFFile(std::unique_ptr<MemoryBuffer> mb, HexagonLinkingContext &ctx)
      : ELFFile<ELFT>(std::move(mb), ctx) {}

  static ErrorOr<std::unique_ptr<HexagonELFFile>>
  create(std::unique_ptr<MemoryBuffer> mb, HexagonLinkingContext &ctx) {
    return std::unique_ptr<HexagonELFFile<ELFT>>(
        new HexagonELFFile<ELFT>(std::move(mb), ctx));
  }

  bool isCommonSymbol(const Elf_Sym *symbol) const override {
    switch (symbol->st_shndx) {
    // Common symbols
    case llvm::ELF::SHN_HEXAGON_SCOMMON:
    case llvm::ELF::SHN_HEXAGON_SCOMMON_1:
    case llvm::ELF::SHN_HEXAGON_SCOMMON_2:
    case llvm::ELF::SHN_HEXAGON_SCOMMON_4:
    case llvm::ELF::SHN_HEXAGON_SCOMMON_8:
      return true;
    default:
      break;
    }
    return ELFFile<ELFT>::isCommonSymbol(symbol);
  }

  /// Process the Defined symbol and create an atom for it.
  ErrorOr<ELFDefinedAtom<ELFT> *>
  handleDefinedSymbol(StringRef symName, StringRef sectionName,
                      const Elf_Sym *sym, const Elf_Shdr *sectionHdr,
                      ArrayRef<uint8_t> contentData,
                      unsigned int referenceStart, unsigned int referenceEnd,
                      std::vector<ELFReference<ELFT> *> &referenceList) override {
    return new (this->_readerStorage) HexagonELFDefinedAtom<ELFT>(
        *this, symName, sectionName, sym, sectionHdr, contentData,
        referenceStart, referenceEnd, referenceList);
  }

  /// Process the Common symbol and create an atom for it.
  ErrorOr<ELFCommonAtom<ELFT> *>
  handleCommonSymbol(StringRef symName, const Elf_Sym *sym) override {
    return new (this->_readerStorage)
        HexagonELFCommonAtom<ELFT>(*this, symName, sym);
  }
};

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_HEXAGON_ELF_FILE_H
