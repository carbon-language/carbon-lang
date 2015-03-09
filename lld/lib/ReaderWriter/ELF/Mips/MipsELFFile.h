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
#include "MipsRelocationHandler.h"

namespace llvm {
namespace object {

template <class ELFT>
struct Elf_RegInfo;

template <llvm::support::endianness TargetEndianness, std::size_t MaxAlign>
struct Elf_RegInfo<ELFType<TargetEndianness, MaxAlign, false>> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, MaxAlign, false)
  Elf_Word ri_gprmask;     // bit-mask of used general registers
  Elf_Word ri_cprmask[4];  // bit-mask of used co-processor registers
  Elf_Addr ri_gp_value;    // gp register value
};

template <llvm::support::endianness TargetEndianness, std::size_t MaxAlign>
struct Elf_RegInfo<ELFType<TargetEndianness, MaxAlign, true>> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, MaxAlign, true)
  Elf_Word ri_gprmask;     // bit-mask of used general registers
  Elf_Word ri_pad;         // unused padding field
  Elf_Word ri_cprmask[4];  // bit-mask of used co-processor registers
  Elf_Addr ri_gp_value;    // gp register value
};

template <class ELFT> struct Elf_Mips_Options {
  LLVM_ELF_IMPORT_TYPES(ELFT::TargetEndianness, ELFT::MaxAlignment,
                        ELFT::Is64Bits)
  uint8_t kind;     // Determines interpretation of variable part of descriptor
  uint8_t size;     // Byte size of descriptor, including this header
  Elf_Half section; // Section header index of section affected,
                    // or 0 for global options
  Elf_Word info;    // Kind-specific information
};

} // end namespace object.
} // end namespace llvm.

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

template <class ELFT> class MipsELFFile : public ELFFile<ELFT> {
public:
  MipsELFFile(std::unique_ptr<MemoryBuffer> mb, MipsLinkingContext &ctx)
      : ELFFile<ELFT>(std::move(mb), ctx) {}

  static ErrorOr<std::unique_ptr<MipsELFFile>>
  create(std::unique_ptr<MemoryBuffer> mb, MipsLinkingContext &ctx) {
    return std::unique_ptr<MipsELFFile<ELFT>>(
        new MipsELFFile<ELFT>(std::move(mb), ctx));
  }

  bool isPIC() const {
    return this->_objFile->getHeader()->e_flags & llvm::ELF::EF_MIPS_PIC;
  }

  /// \brief gp register value stored in the .reginfo section.
  int64_t getGP0() const { return _gp0 ? *_gp0 : 0; }

  /// \brief .tdata section address plus fixed offset.
  uint64_t getTPOffset() const { return *_tpOff; }
  uint64_t getDTPOffset() const { return *_dtpOff; }

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

  enum { TP_OFFSET = 0x7000, DTP_OFFSET = 0x8000 };

  llvm::Optional<int64_t> _gp0;
  llvm::Optional<uint64_t> _tpOff;
  llvm::Optional<uint64_t> _dtpOff;

  ErrorOr<ELFDefinedAtom<ELFT> *> handleDefinedSymbol(
      StringRef symName, StringRef sectionName, const Elf_Sym *sym,
      const Elf_Shdr *sectionHdr, ArrayRef<uint8_t> contentData,
      unsigned int referenceStart, unsigned int referenceEnd,
      std::vector<ELFReference<ELFT> *> &referenceList) override {
    return new (this->_readerStorage) MipsELFDefinedAtom<ELFT>(
        *this, symName, sectionName, sym, sectionHdr, contentData,
        referenceStart, referenceEnd, referenceList);
  }

  const Elf_Shdr *findSectionByType(uint64_t type) {
    for (const Elf_Shdr &section : this->_objFile->sections())
      if (section.sh_type == type)
        return &section;
    return nullptr;
  }

  const Elf_Shdr *findSectionByFlags(uint64_t flags) {
    for (const Elf_Shdr &section : this->_objFile->sections())
      if (section.sh_flags & flags)
        return &section;
    return nullptr;
  }

  std::error_code readAuxData() {
    using namespace llvm::ELF;
    if (const Elf_Shdr *sec = findSectionByFlags(SHF_TLS)) {
      _tpOff = sec->sh_addr + TP_OFFSET;
      _dtpOff = sec->sh_addr + DTP_OFFSET;
    }

    typedef llvm::object::Elf_RegInfo<ELFT> Elf_RegInfo;
    typedef llvm::object::Elf_Mips_Options<ELFT> Elf_Mips_Options;

    if (const Elf_Shdr *sec = findSectionByType(SHT_MIPS_OPTIONS)) {
      auto contents = this->getSectionContents(sec);
      if (std::error_code ec = contents.getError())
        return ec;

      ArrayRef<uint8_t> raw = contents.get();
      while (!raw.empty()) {
        if (raw.size() < sizeof(Elf_Mips_Options))
          return make_dynamic_error_code(
              StringRef("Invalid size of MIPS_OPTIONS section"));

        const auto *opt = reinterpret_cast<const Elf_Mips_Options *>(raw.data());
        if (opt->kind == ODK_REGINFO) {
          _gp0 = reinterpret_cast<const Elf_RegInfo *>(opt + 1)->ri_gp_value;
          break;
        }
        raw = raw.slice(opt->size);
      }
    } else if (const Elf_Shdr *sec = findSectionByType(SHT_MIPS_REGINFO)) {
      auto contents = this->getSectionContents(sec);
      if (std::error_code ec = contents.getError())
        return ec;

      ArrayRef<uint8_t> raw = contents.get();
      if (raw.size() != sizeof(Elf_RegInfo))
        return make_dynamic_error_code(
            StringRef("Invalid size of MIPS_REGINFO section"));

      _gp0 = reinterpret_cast<const Elf_RegInfo *>(raw.data())->ri_gp_value;
    }
    return std::error_code();
  }

  void createRelocationReferences(const Elf_Sym *symbol,
                                  ArrayRef<uint8_t> symContent,
                                  ArrayRef<uint8_t> secContent,
                                  range<Elf_Rel_Iter> rels) override {
    for (Elf_Rel_Iter rit = rels.begin(), eit = rels.end(); rit != eit; ++rit) {
      if (rit->r_offset < symbol->st_value ||
          symbol->st_value + symContent.size() <= rit->r_offset)
        continue;

      auto elfReference = new (this->_readerStorage) ELFReference<ELFT>(
          rit->r_offset - symbol->st_value, this->kindArch(),
          rit->getType(isMips64EL()), rit->getSymbol(isMips64EL()));
      ELFFile<ELFT>::addReferenceToSymbol(elfReference, symbol);
      this->_references.push_back(elfReference);

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
    return MipsRelocationHandler<ELFT>::readAddend(
        ri.getType(isMips64EL()), content.data() + ri.r_offset);
  }

  uint32_t getPairRelocation(const Elf_Rel &rel) const {
    switch (rel.getType(isMips64EL())) {
    case llvm::ELF::R_MIPS_HI16:
      return llvm::ELF::R_MIPS_LO16;
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
      return rel.getType(isMips64EL()) == pairRelType &&
             rel.getSymbol(isMips64EL()) == rit->getSymbol(isMips64EL());
    });
  }

  bool isMips64EL() const { return this->_objFile->isMips64EL(); }
  bool isLocalBinding(const Elf_Rel &rel) const {
    return this->_objFile->getSymbol(rel.getSymbol(isMips64EL()))
               ->getBinding() == llvm::ELF::STB_LOCAL;
  }
};

template <class ELFT> class MipsDynamicFile : public DynamicFile<ELFT> {
public:
  MipsDynamicFile(const MipsLinkingContext &context, StringRef name)
      : DynamicFile<ELFT>(context, name) {}
};

} // elf
} // lld

#endif
