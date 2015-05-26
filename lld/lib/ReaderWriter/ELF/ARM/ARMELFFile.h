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

class ARMELFBaseDefinedAtom : public ELFDefinedAtom<ELF32LE> {
public:
  /// The values of custom content type enum must not interfere
  /// with ones in base defined atom class' enum.
  enum ARMContentType {
    typeARMExidx = 0x1000,   // Identifies ARM_EXIDX section
  };

  template <typename... T>
  ARMELFBaseDefinedAtom(T &&... args)
      : ELFDefinedAtom<ELF32LE>(std::forward<T>(args)...) {}

  DefinedAtom::ContentPermissions permissions() const override {
    if (_permissions != DefinedAtom::permUnknown)
      return _permissions;

    switch (_section->sh_type) {
    case llvm::ELF::SHT_ARM_EXIDX:
      return _permissions = permR__;
    }
    return ELFDefinedAtom::permissions();
  }

  DefinedAtom::ContentType contentType() const override {
    if (_contentType != DefinedAtom::typeUnknown)
      return _contentType;

    switch (_section->sh_type) {
    case llvm::ELF::SHT_ARM_EXIDX:
      return _contentType = (DefinedAtom::ContentType)typeARMExidx;
    }
    return ELFDefinedAtom::contentType();
  }
};

class ARMELFMappingAtom : public ARMELFBaseDefinedAtom {
public:
  template <typename... T>
  ARMELFMappingAtom(DefinedAtom::CodeModel model, T &&... args)
      : ARMELFBaseDefinedAtom(std::forward<T>(args)...), _model(model) {}

  DefinedAtom::CodeModel codeModel() const override { return _model; }

private:
  DefinedAtom::CodeModel _model;
};

class ARMELFDefinedAtom : public ARMELFBaseDefinedAtom {
public:
  template <typename... T>
  ARMELFDefinedAtom(T &&... args)
      : ARMELFBaseDefinedAtom(std::forward<T>(args)...) {}

  bool isThumbFunc() const {
    const auto *symbol = _symbol;
    return symbol->getType() == llvm::ELF::STT_FUNC &&
           (static_cast<uint64_t>(symbol->st_value) & 0x1);
  }

  /// Correct st_value for symbols addressing Thumb instructions
  /// by removing its zero bit.
  uint64_t getSymbolValue() const override {
    const auto value = static_cast<uint64_t>(_symbol->st_value);
    return isThumbFunc() ? value & ~0x1 : value;
  }

  DefinedAtom::CodeModel codeModel() const override {
    return isThumbFunc() ? DefinedAtom::codeARMThumb : DefinedAtom::codeNA;
  }
};

class ARMELFFile : public ELFFile<ELF32LE> {
  typedef llvm::object::Elf_Rel_Impl<ELF32LE, false> Elf_Rel;

public:
  ARMELFFile(std::unique_ptr<MemoryBuffer> mb, ELFLinkingContext &ctx)
      : ELFFile(std::move(mb), ctx) {}

protected:
  /// Returns initial addend; for ARM it is 0, because it is read
  /// during the relocations applying
  Reference::Addend getInitialAddend(ArrayRef<uint8_t>, uint64_t,
                                     const Elf_Rel &) const override {
    return 0;
  }

private:
  typedef llvm::object::Elf_Sym_Impl<ELF32LE> Elf_Sym;
  typedef llvm::object::Elf_Shdr_Impl<ELF32LE> Elf_Shdr;

  /// Correct st_value for symbols addressing Thumb instructions
  /// by removing its zero bit.
  uint64_t getSymbolValue(const Elf_Sym *symbol) const override {
    const auto value = static_cast<uint64_t>(symbol->st_value);
    return symbol->getType() == llvm::ELF::STT_FUNC ? value & ~0x1 : value;
  }

  /// Process the Defined symbol and create an atom for it.
  ELFDefinedAtom<ELF32LE> *createDefinedAtom(
      StringRef symName, StringRef sectionName, const Elf_Sym *sym,
      const Elf_Shdr *sectionHdr, ArrayRef<uint8_t> contentData,
      unsigned int referenceStart, unsigned int referenceEnd,
      std::vector<ELFReference<ELF32LE> *> &referenceList) override {
    if (symName.size() >= 2 && symName[0] == '$') {
      switch (symName[1]) {
      case 'a':
        return new (_readerStorage)
            ARMELFMappingAtom(DefinedAtom::codeARM_a, *this, symName,
                              sectionName, sym, sectionHdr, contentData,
                              referenceStart, referenceEnd, referenceList);
      case 'd':
        return new (_readerStorage)
            ARMELFMappingAtom(DefinedAtom::codeARM_d, *this, symName,
                              sectionName, sym, sectionHdr, contentData,
                              referenceStart, referenceEnd, referenceList);
      case 't':
        return new (_readerStorage)
            ARMELFMappingAtom(DefinedAtom::codeARM_t, *this, symName,
                              sectionName, sym, sectionHdr, contentData,
                              referenceStart, referenceEnd, referenceList);
      default:
        // Fall through and create regular defined atom.
        break;
      }
    }
    return new (_readerStorage) ARMELFDefinedAtom(
        *this, symName, sectionName, sym, sectionHdr, contentData,
        referenceStart, referenceEnd, referenceList);
  }
};

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_ARM_ARM_ELF_FILE_H
