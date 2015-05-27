//===- lib/ReaderWriter/ELF/MipsELFFile.h ---------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_MIPS_MIPS_ELF_FILE_H
#define LLD_READER_WRITER_ELF_MIPS_MIPS_ELF_FILE_H

#include "ELFReader.h"
#include "MipsLinkingContext.h"
#include "MipsReginfo.h"
#include "MipsRelocationHandler.h"
#include "llvm/ADT/STLExtras.h"

namespace lld {
namespace elf {

template <class ELFT> class MipsELFFile;

template <class ELFT>
class MipsELFDefinedAtom : public ELFDefinedAtom<ELFT> {
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;
  typedef llvm::object::Elf_Shdr_Impl<ELFT> Elf_Shdr;

public:
  MipsELFDefinedAtom(const MipsELFFile<ELFT> &file, StringRef symbolName,
                     StringRef sectionName, const Elf_Sym *symbol,
                     const Elf_Shdr *section, ArrayRef<uint8_t> contentData,
                     unsigned int referenceStart, unsigned int referenceEnd,
                     std::vector<ELFReference<ELFT> *> &referenceList)
      : ELFDefinedAtom<ELFT>(file, symbolName, sectionName, symbol, section,
                             contentData, referenceStart, referenceEnd,
                             referenceList) {}

  const MipsELFFile<ELFT>& file() const override {
    return static_cast<const MipsELFFile<ELFT> &>(this->_owningFile);
  }

  DefinedAtom::CodeModel codeModel() const override {
    switch (this->_symbol->st_other & llvm::ELF::STO_MIPS_MIPS16) {
    case llvm::ELF::STO_MIPS_MIPS16:
      return DefinedAtom::codeMips16;
    case llvm::ELF::STO_MIPS_PIC:
      return DefinedAtom::codeMipsPIC;
    case llvm::ELF::STO_MIPS_MICROMIPS:
      return DefinedAtom::codeMipsMicro;
    case llvm::ELF::STO_MIPS_MICROMIPS | llvm::ELF::STO_MIPS_PIC:
      return DefinedAtom::codeMipsMicroPIC;
    default:
      return DefinedAtom::codeNA;
    }
  }
};

template <class ELFT> class MipsELFReference : public ELFReference<ELFT> {
  typedef llvm::object::Elf_Rel_Impl<ELFT, false> Elf_Rel;
  typedef llvm::object::Elf_Rel_Impl<ELFT, true> Elf_Rela;

  static const bool _isMips64EL =
      ELFT::Is64Bits && ELFT::TargetEndianness == llvm::support::little;

public:
  MipsELFReference(uint64_t symValue, const Elf_Rela &rel)
      : ELFReference<ELFT>(
            &rel, rel.r_offset - symValue, Reference::KindArch::Mips,
            rel.getType(_isMips64EL) & 0xff, rel.getSymbol(_isMips64EL)),
        _tag(extractTag(rel)) {}

  MipsELFReference(uint64_t symValue, const Elf_Rel &rel)
      : ELFReference<ELFT>(rel.r_offset - symValue, Reference::KindArch::Mips,
                           rel.getType(_isMips64EL) & 0xff,
                           rel.getSymbol(_isMips64EL)),
        _tag(extractTag(rel)) {}

  uint32_t tag() const override { return _tag; }
  void setTag(uint32_t tag) { _tag = tag; }

private:
  uint32_t _tag;

  template <class R> static uint32_t extractTag(const R &rel) {
    return (rel.getType(_isMips64EL) & 0xffffff00) >> 8;
  }
};

template <class ELFT> class MipsELFFile : public ELFFile<ELFT> {
public:
  MipsELFFile(std::unique_ptr<MemoryBuffer> mb, ELFLinkingContext &ctx)
      : ELFFile<ELFT>(std::move(mb), ctx) {}

  bool isPIC() const {
    return this->_objFile->getHeader()->e_flags & llvm::ELF::EF_MIPS_PIC;
  }

  /// \brief gp register value stored in the .reginfo section.
  int64_t getGP0() const { return _gp0; }

  /// \brief .tdata section address plus fixed offset.
  uint64_t getTPOffset() const { return _tpOff; }
  uint64_t getDTPOffset() const { return _dtpOff; }

protected:
  std::error_code doParse() override {
    if (std::error_code ec = ELFFile<ELFT>::doParse())
      return ec;
    // Retrieve some auxiliary data like GP value, TLS section address etc
    // from the object file.
    return readAuxData();
  }

private:
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;
  typedef llvm::object::Elf_Shdr_Impl<ELFT> Elf_Shdr;
  typedef llvm::object::Elf_Rel_Impl<ELFT, false> Elf_Rel;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rel_Iter Elf_Rel_Iter;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Rela_Iter Elf_Rela_Iter;

  enum { TP_OFFSET = 0x7000, DTP_OFFSET = 0x8000 };

  static const bool _isMips64EL =
      ELFT::Is64Bits && ELFT::TargetEndianness == llvm::support::little;

  int64_t _gp0 = 0;
  uint64_t _tpOff = 0;
  uint64_t _dtpOff = 0;

  ELFDefinedAtom<ELFT> *
  createDefinedAtom(StringRef symName, StringRef sectionName,
                    const Elf_Sym *sym, const Elf_Shdr *sectionHdr,
                    ArrayRef<uint8_t> contentData, unsigned int referenceStart,
                    unsigned int referenceEnd,
                    std::vector<ELFReference<ELFT> *> &referenceList) override {
    return new (this->_readerStorage) MipsELFDefinedAtom<ELFT>(
        *this, symName, sectionName, sym, sectionHdr, contentData,
        referenceStart, referenceEnd, referenceList);
  }

  const Elf_Shdr *findSectionByType(uint64_t type) const {
    for (const Elf_Shdr &section : this->_objFile->sections())
      if (section.sh_type == type)
        return &section;
    return nullptr;
  }

  const Elf_Shdr *findSectionByFlags(uint64_t flags) const {
    for (const Elf_Shdr &section : this->_objFile->sections())
      if (section.sh_flags & flags)
        return &section;
    return nullptr;
  }

  typedef typename llvm::object::ELFFile<ELFT>::Elf_Ehdr Elf_Ehdr;
  typedef llvm::object::Elf_Mips_RegInfo<ELFT> Elf_Mips_RegInfo;
  typedef llvm::object::Elf_Mips_Options<ELFT> Elf_Mips_Options;

  ErrorOr<const Elf_Mips_RegInfo *> findRegInfoSec() const {
    using namespace llvm::ELF;
    if (const Elf_Shdr *sec = findSectionByType(SHT_MIPS_OPTIONS)) {
      auto contents = this->getSectionContents(sec);
      if (std::error_code ec = contents.getError())
        return ec;

      ArrayRef<uint8_t> raw = contents.get();
      while (!raw.empty()) {
        if (raw.size() < sizeof(Elf_Mips_Options))
          return make_dynamic_error_code(
              StringRef("Invalid size of MIPS_OPTIONS section"));

        const auto *opt =
            reinterpret_cast<const Elf_Mips_Options *>(raw.data());
        if (opt->kind == ODK_REGINFO)
          return &opt->getRegInfo();
        raw = raw.slice(opt->size);
      }
    } else if (const Elf_Shdr *sec = findSectionByType(SHT_MIPS_REGINFO)) {
      auto contents = this->getSectionContents(sec);
      if (std::error_code ec = contents.getError())
        return ec;

      ArrayRef<uint8_t> raw = contents.get();
      if (raw.size() != sizeof(Elf_Mips_RegInfo))
        return make_dynamic_error_code(
            StringRef("Invalid size of MIPS_REGINFO section"));

      return reinterpret_cast<const Elf_Mips_RegInfo *>(raw.data());
    }
    return nullptr;
  }

  std::error_code readAuxData() {
    using namespace llvm::ELF;
    if (const Elf_Shdr *sec = findSectionByFlags(SHF_TLS)) {
      _tpOff = sec->sh_addr + TP_OFFSET;
      _dtpOff = sec->sh_addr + DTP_OFFSET;
    }

    auto &ctx = static_cast<MipsLinkingContext &>(this->_ctx);

    ErrorOr<const Elf_Mips_RegInfo *> regInfoSec = findRegInfoSec();
    if (auto ec = regInfoSec.getError())
      return ec;
    if (const Elf_Mips_RegInfo *regInfo = regInfoSec.get()) {
      ctx.mergeReginfoMask(*regInfo);
      _gp0 = regInfo->ri_gp_value;
    }

    const Elf_Ehdr *hdr = this->_objFile->getHeader();
    if (std::error_code ec = ctx.mergeElfFlags(hdr->e_flags))
      return ec;

    return std::error_code();
  }

  void createRelocationReferences(const Elf_Sym *symbol,
                                  ArrayRef<uint8_t> content,
                                  range<Elf_Rela_Iter> rels) override {
    const auto value = this->getSymbolValue(symbol);
    for (const auto &rel : rels) {
      if (rel.r_offset < value || value + content.size() <= rel.r_offset)
        continue;
      auto r = new (this->_readerStorage) MipsELFReference<ELFT>(value, rel);
      this->addReferenceToSymbol(r, symbol);
      this->_references.push_back(r);
    }
  }

  void createRelocationReferences(const Elf_Sym *symbol,
                                  ArrayRef<uint8_t> symContent,
                                  ArrayRef<uint8_t> secContent,
                                  range<Elf_Rel_Iter> rels) override {
    const auto value = this->getSymbolValue(symbol);
    for (Elf_Rel_Iter rit = rels.begin(), eit = rels.end(); rit != eit; ++rit) {
      if (rit->r_offset < value || value + symContent.size() <= rit->r_offset)
        continue;

      auto r = new (this->_readerStorage) MipsELFReference<ELFT>(value, *rit);
      this->addReferenceToSymbol(r, symbol);
      this->_references.push_back(r);

      auto addend = readAddend(*rit, secContent);
      auto pairRelType = getPairRelocation(*rit);
      if (pairRelType != llvm::ELF::R_MIPS_NONE) {
        addend <<= 16;
        auto mit = findMatchingRelocation(pairRelType, rit, eit);
        if (mit != eit)
          addend += int16_t(readAddend(*mit, secContent));
        else
          // FIXME (simon): Show detailed warning.
          llvm::errs() << "lld warning: cannot matching LO16 relocation\n";
      }
      this->_references.back()->setAddend(addend);
    }
  }

  Reference::Addend readAddend(const Elf_Rel &ri,
                               const ArrayRef<uint8_t> content) const {
    return readMipsRelocAddend(getPrimaryType(ri),
                               content.data() + ri.r_offset);
  }

  uint32_t getPairRelocation(const Elf_Rel &rel) const {
    switch (getPrimaryType(rel)) {
    case llvm::ELF::R_MIPS_HI16:
      return llvm::ELF::R_MIPS_LO16;
    case llvm::ELF::R_MIPS_PCHI16:
      return llvm::ELF::R_MIPS_PCLO16;
    case llvm::ELF::R_MIPS_GOT16:
      if (isLocalBinding(rel))
        return llvm::ELF::R_MIPS_LO16;
      break;
    case llvm::ELF::R_MICROMIPS_HI16:
      return llvm::ELF::R_MICROMIPS_LO16;
    case llvm::ELF::R_MICROMIPS_GOT16:
      if (isLocalBinding(rel))
        return llvm::ELF::R_MICROMIPS_LO16;
      break;
    default:
      // Nothing to do.
      break;
    }
    return llvm::ELF::R_MIPS_NONE;
  }

  Elf_Rel_Iter findMatchingRelocation(uint32_t pairRelType, Elf_Rel_Iter rit,
                                      Elf_Rel_Iter eit) const {
    return std::find_if(rit, eit, [&](const Elf_Rel &rel) {
      return getPrimaryType(rel) == pairRelType &&
             rel.getSymbol(_isMips64EL) == rit->getSymbol(_isMips64EL);
    });
  }

  static uint8_t getPrimaryType(const Elf_Rel &rel) {
    return rel.getType(_isMips64EL) & 0xff;
  }
  bool isLocalBinding(const Elf_Rel &rel) const {
    return this->_objFile->getSymbol(rel.getSymbol(_isMips64EL))
               ->getBinding() == llvm::ELF::STB_LOCAL;
  }
};

} // elf
} // lld

#endif
