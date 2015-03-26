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

template <class ELFT, DefinedAtom::CodeModel Model>
class ARMELFMappingAtom : public ELFDefinedAtom<ELFT> {
public:
  template<typename... T>
  ARMELFMappingAtom(T&&... args)
      : ELFDefinedAtom<ELFT>(std::forward<T>(args)...) {}

  DefinedAtom::CodeModel codeModel() const override {
    return Model;
  }
};

template <class ELFT> class ARMELFDefinedAtom : public ELFDefinedAtom<ELFT> {
public:
  template<typename... T>
  ARMELFDefinedAtom(T&&... args)
      : ELFDefinedAtom<ELFT>(std::forward<T>(args)...) {}

  bool isThumbFunc() const {
    const auto* symbol = this->_symbol;
    return symbol->getType() == llvm::ELF::STT_FUNC &&
        (static_cast<uint64_t>(symbol->st_value) & 0x1);
  }

  /// Correct st_value for symbols addressing Thumb instructions
  /// by removing its zero bit.
  uint64_t getSymbolValue() const override {
    const auto value = static_cast<uint64_t>(this->_symbol->st_value);
    return isThumbFunc() ? value & ~0x1 : value;
  }

  DefinedAtom::CodeModel codeModel() const override {
    return isThumbFunc() ? DefinedAtom::codeARMThumb : DefinedAtom::codeNA;
  }
};

template <class ELFT> class ARMELFFile : public ELFFile<ELFT> {
  typedef llvm::object::Elf_Rel_Impl<ELFT, false> Elf_Rel;

public:
  ARMELFFile(std::unique_ptr<MemoryBuffer> mb, ARMLinkingContext &ctx)
      : ELFFile<ELFT>(std::move(mb), ctx) {}

  static ErrorOr<std::unique_ptr<ARMELFFile>>
  create(std::unique_ptr<MemoryBuffer> mb, ARMLinkingContext &ctx) {
    return std::unique_ptr<ARMELFFile<ELFT>>(
        new ARMELFFile<ELFT>(std::move(mb), ctx));
  }

protected:
  /// Returns initial addend; for ARM it is 0, because it is read
  /// during the relocations applying
  Reference::Addend getInitialAddend(ArrayRef<uint8_t>,
                                     uint64_t,
                                     const Elf_Rel&) const override {
    return 0;
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
    if (symName.size() >= 2 && symName[0] == '$') {
      switch (symName[1]) {
      case 'a':
        return new (this->_readerStorage)
            ARMELFMappingAtom<ELFT, DefinedAtom::codeARM_a>(
                *this, symName, sectionName, sym, sectionHdr, contentData,
                referenceStart, referenceEnd, referenceList);
      case 'd':
        return new (this->_readerStorage)
            ARMELFMappingAtom<ELFT, DefinedAtom::codeARM_d>(
                *this, symName, sectionName, sym, sectionHdr, contentData,
                referenceStart, referenceEnd, referenceList);
      case 't':
        return new (this->_readerStorage)
            ARMELFMappingAtom<ELFT, DefinedAtom::codeARM_t>(
                *this, symName, sectionName, sym, sectionHdr, contentData,
                referenceStart, referenceEnd, referenceList);
      default:
        // Fall through and create regular defined atom.
      break;
      }
    }
    return new (this->_readerStorage) ARMELFDefinedAtom<ELFT>(
        *this, symName, sectionName, sym, sectionHdr, contentData,
        referenceStart, referenceEnd, referenceList);
  }
};

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_ARM_ARM_ELF_FILE_H
