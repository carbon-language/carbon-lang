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

class HexagonELFFile;

class HexagonELFDefinedAtom : public ELFDefinedAtom<ELF32LE> {
  typedef llvm::object::Elf_Sym_Impl<ELF32LE> Elf_Sym;
  typedef llvm::object::Elf_Shdr_Impl<ELF32LE> Elf_Shdr;

public:
  template <typename... T>
  HexagonELFDefinedAtom(T &&... args)
      : ELFDefinedAtom(std::forward<T>(args)...) {}

  DefinedAtom::ContentType contentType() const override {
    if (_contentType != DefinedAtom::typeUnknown)
      return _contentType;
    if (_section->sh_flags & llvm::ELF::SHF_HEX_GPREL) {
      if (_section->sh_type == llvm::ELF::SHT_NOBITS)
        return (_contentType = DefinedAtom::typeZeroFillFast);
      return (_contentType = DefinedAtom::typeDataFast);
    }
    return ELFDefinedAtom::contentType();
  }

  DefinedAtom::ContentPermissions permissions() const override {
    if (_section->sh_flags & llvm::ELF::SHF_HEX_GPREL)
      return DefinedAtom::permRW_;
    return ELFDefinedAtom::permissions();
  }
};

class HexagonELFCommonAtom : public ELFCommonAtom<ELF32LE> {
  typedef llvm::object::Elf_Sym_Impl<ELF32LE> Elf_Sym;
  typedef llvm::object::Elf_Shdr_Impl<ELF32LE> Elf_Shdr;

public:
  HexagonELFCommonAtom(const ELFFile<ELF32LE> &file, StringRef symbolName,
                       const Elf_Sym *symbol)
      : ELFCommonAtom(file, symbolName, symbol) {}

  virtual bool isSmallCommonSymbol() const {
    switch (_symbol->st_shndx) {
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

  uint64_t size() const override {
    if (isSmallCommonSymbol())
      return _symbol->st_size;
    return ELFCommonAtom::size();
  }

  DefinedAtom::Merge merge() const override {
    if (_symbol->getBinding() == llvm::ELF::STB_WEAK)
      return DefinedAtom::mergeAsWeak;
    if (isSmallCommonSymbol())
      return DefinedAtom::mergeAsTentative;
    return ELFCommonAtom::merge();
  }

  DefinedAtom::ContentType contentType() const override {
    if (isSmallCommonSymbol())
      return DefinedAtom::typeZeroFillFast;
    return ELFCommonAtom::contentType();
  }

  DefinedAtom::Alignment alignment() const override {
    if (isSmallCommonSymbol())
      return DefinedAtom::Alignment(_symbol->st_value);
    return 1;
  }

  DefinedAtom::ContentPermissions permissions() const override {
    if (isSmallCommonSymbol())
      return DefinedAtom::permRW_;
    return ELFCommonAtom::permissions();
  }
};

class HexagonELFFile : public ELFFile<ELF32LE> {
  typedef llvm::object::Elf_Sym_Impl<ELF32LE> Elf_Sym;
  typedef llvm::object::Elf_Shdr_Impl<ELF32LE> Elf_Shdr;

public:
  HexagonELFFile(std::unique_ptr<MemoryBuffer> mb, ELFLinkingContext &ctx)
      : ELFFile(std::move(mb), ctx) {}

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
    return ELFFile::isCommonSymbol(symbol);
  }

  /// Process the Defined symbol and create an atom for it.
  ELFDefinedAtom<ELF32LE> *createDefinedAtom(
      StringRef symName, StringRef sectionName, const Elf_Sym *sym,
      const Elf_Shdr *sectionHdr, ArrayRef<uint8_t> contentData,
      unsigned int referenceStart, unsigned int referenceEnd,
      std::vector<ELFReference<ELF32LE> *> &referenceList) override {
    return new (_readerStorage) HexagonELFDefinedAtom(
        *this, symName, sectionName, sym, sectionHdr, contentData,
        referenceStart, referenceEnd, referenceList);
  }

  /// Process the Common symbol and create an atom for it.
  ELFCommonAtom<ELF32LE> *createCommonAtom(StringRef symName,
                                           const Elf_Sym *sym) override {
    return new (_readerStorage) HexagonELFCommonAtom(*this, symName, sym);
  }
};

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_HEXAGON_ELF_FILE_H
