//===--------- lib/ReaderWriter/ELF/ARM/ARMELFFile.h ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_ARM_ARM_ELF_FILE_H
#define LLD_READER_WRITER_ELF_ARM_ARM_ELF_FILE_H

#include "ELFReader.h"

namespace lld {
namespace elf {

class ARMLinkingContext;

template <class ELFT> class ARMELFDefinedAtom : public ELFDefinedAtom<ELFT> {
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;
  typedef llvm::object::Elf_Shdr_Impl<ELFT> Elf_Shdr;

public:
  ARMELFDefinedAtom(const ELFFile<ELFT> &file, StringRef symbolName,
                 StringRef sectionName, const Elf_Sym *symbol,
                 const Elf_Shdr *section, ArrayRef<uint8_t> contentData,
                 unsigned int referenceStart, unsigned int referenceEnd,
                 std::vector<ELFReference<ELFT> *> &referenceList)
      : ELFDefinedAtom<ELFT>(file, symbolName, sectionName, symbol, section,
                             contentData, referenceStart, referenceEnd,
                             referenceList) {}

  bool isThumbFunc(const Elf_Sym *symbol) const {
    return symbol->getType() == llvm::ELF::STT_FUNC &&
        (static_cast<uint64_t>(symbol->st_value) & 0x1);
  }

  /// Correct st_value for symbols addressing Thumb instructions
  /// by removing its zero bit.
  uint64_t getSymbolValue(const Elf_Sym *symbol) const override {
    const auto value = static_cast<uint64_t>(symbol->st_value);
    return isThumbFunc(symbol) ? value & ~0x1 : value;
  }

  DefinedAtom::CodeModel codeModel() const override {
    if (isThumbFunc(this->_symbol))
      return DefinedAtom::codeARMThumb;
    return DefinedAtom::codeNA;
  }
};

template <class ELFT> class ARMELFFile : public ELFFile<ELFT> {
public:
  ARMELFFile(std::unique_ptr<MemoryBuffer> mb, ARMLinkingContext &ctx)
      : ELFFile<ELFT>(std::move(mb), ctx) {}

  static ErrorOr<std::unique_ptr<ARMELFFile>>
  create(std::unique_ptr<MemoryBuffer> mb, ARMLinkingContext &ctx) {
    return std::unique_ptr<ARMELFFile<ELFT>>(
        new ARMELFFile<ELFT>(std::move(mb), ctx));
  }

private:
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;
  typedef llvm::object::Elf_Shdr_Impl<ELFT> Elf_Shdr;

  /// Correct st_value for symbols addressing Thumb instructions
  /// by removing its zero bit.
  uint64_t getSymbolValue(const Elf_Sym *symbol) const override {
    const auto value = static_cast<uint64_t>(symbol->st_value);
    return symbol->getType() == llvm::ELF::STT_FUNC ? value & ~0x1 : value;
  }

  /// Process the Defined symbol and create an atom for it.
  ErrorOr<ELFDefinedAtom<ELFT> *> handleDefinedSymbol(StringRef symName,
          StringRef sectionName,
          const Elf_Sym *sym, const Elf_Shdr *sectionHdr,
          ArrayRef<uint8_t> contentData,
          unsigned int referenceStart, unsigned int referenceEnd,
          std::vector<ELFReference<ELFT> *> &referenceList) override {
    return new (this->_readerStorage) ARMELFDefinedAtom<ELFT>(
        *this, symName, sectionName, sym, sectionHdr, contentData,
        referenceStart, referenceEnd, referenceList);
  }
};

template <class ELFT> class ARMDynamicFile : public DynamicFile<ELFT> {
public:
  ARMDynamicFile(const ARMLinkingContext &context, StringRef name)
      : DynamicFile<ELFT>(context, name) {}
};

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_ARM_ARM_ELF_FILE_H
