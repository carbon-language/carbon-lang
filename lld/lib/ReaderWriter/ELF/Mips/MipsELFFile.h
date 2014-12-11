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
#include "llvm/Support/Endian.h"

namespace llvm {
namespace object {

template <class ELFT>
struct Elf_RegInfo;

template <llvm::support::endianness TargetEndianness, std::size_t MaxAlign>
struct Elf_RegInfo<ELFType<TargetEndianness, MaxAlign, false>> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, MaxAlign, false)
  Elf_Word ri_gprmask;     // bit-mask of used general registers
  Elf_Word ri_cprmask[4];  // bit-mask of used co-processor registers
  Elf_Sword ri_gp_value;   // gp register value
};

template <llvm::support::endianness TargetEndianness, std::size_t MaxAlign>
struct Elf_RegInfo<ELFType<TargetEndianness, MaxAlign, true>> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, MaxAlign, true)
  Elf_Word ri_gprmask;     // bit-mask of used general registers
  Elf_Word ri_cprmask[4];  // bit-mask of used co-processor registers
  Elf_Sword ri_gp_value;   // gp register value
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
  MipsELFFile(std::unique_ptr<MemoryBuffer> mb, bool atomizeStrings)
      : ELFFile<ELFT>(std::move(mb), atomizeStrings) {}

  static ErrorOr<std::unique_ptr<MipsELFFile>>
  create(std::unique_ptr<MemoryBuffer> mb, bool atomizeStrings) {
    std::unique_ptr<MipsELFFile<ELFT>> file(
        new MipsELFFile<ELFT>(std::move(mb), atomizeStrings));
    if (std::error_code ec = file->parse())
      return ec;
    return std::move(file);
  }

  bool isPIC() const {
    return this->_objFile->getHeader()->e_flags & llvm::ELF::EF_MIPS_PIC;
  }

  /// \brief gp register value stored in the .reginfo section.
  int64_t getGP0() const { return *_gp0; }

  /// \brief .tdata section address plus fixed offset.
  uint64_t getTPOffset() const { return *_tpOff; }
  uint64_t getDTPOffset() const { return *_dtpOff; }

  std::error_code parse() override {
    if (std::error_code ec = ELFFile<ELFT>::parse())
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

  std::error_code readAuxData() {
    typedef llvm::object::Elf_RegInfo<ELFT> Elf_RegInfo;

    for (const Elf_Shdr &section : this->_objFile->sections()) {
      if (!_gp0.hasValue() && section.sh_type == llvm::ELF::SHT_MIPS_REGINFO) {
        auto contents = this->getSectionContents(&section);
        if (std::error_code ec = contents.getError())
          return ec;

        ArrayRef<uint8_t> raw = contents.get();

        // FIXME (simon): Show error in case of invalid section size.
        assert(raw.size() == sizeof(Elf_RegInfo) &&
               "Invalid size of RegInfo section");
        _gp0 = reinterpret_cast<const Elf_RegInfo *>(raw.data())->ri_gp_value;
      } else if (!_tpOff.hasValue() && section.sh_flags & llvm::ELF::SHF_TLS) {
        _tpOff = section.sh_addr + TP_OFFSET;
        _dtpOff = section.sh_addr + DTP_OFFSET;
      }
    }
    return std::error_code();
  }

  void createRelocationReferences(const Elf_Sym &symbol,
                                  ArrayRef<uint8_t> symContent,
                                  ArrayRef<uint8_t> secContent,
                                  range<Elf_Rel_Iter> rels) override {
    for (Elf_Rel_Iter rit = rels.begin(), eit = rels.end(); rit != eit; ++rit) {
      if (rit->r_offset < symbol.st_value ||
          symbol.st_value + symContent.size() <= rit->r_offset)
        continue;

      this->_references.push_back(new (this->_readerStorage) ELFReference<ELFT>(
          &*rit, rit->r_offset - symbol.st_value, this->kindArch(),
          rit->getType(isMips64EL()), rit->getSymbol(isMips64EL())));

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
    using namespace llvm::support;
    const uint8_t *ap = content.data() + ri.r_offset;
    switch (ri.getType(isMips64EL())) {
    case llvm::ELF::R_MIPS_32:
    case llvm::ELF::R_MIPS_GPREL32:
    case llvm::ELF::R_MIPS_PC32:
      return endian::read<int32_t, ELFT::TargetEndianness, 2>(ap);
    case llvm::ELF::R_MIPS_26:
      return endian::read<int32_t, ELFT::TargetEndianness, 2>(ap) & 0x3ffffff;
    case llvm::ELF::R_MIPS_HI16:
    case llvm::ELF::R_MIPS_LO16:
    case llvm::ELF::R_MIPS_GOT16:
    case llvm::ELF::R_MIPS_TLS_DTPREL_HI16:
    case llvm::ELF::R_MIPS_TLS_DTPREL_LO16:
    case llvm::ELF::R_MIPS_TLS_TPREL_HI16:
    case llvm::ELF::R_MIPS_TLS_TPREL_LO16:
      return endian::read<int16_t, ELFT::TargetEndianness, 2>(ap);
    case llvm::ELF::R_MIPS_CALL16:
    case llvm::ELF::R_MIPS_TLS_GD:
    case llvm::ELF::R_MIPS_TLS_LDM:
    case llvm::ELF::R_MIPS_TLS_GOTTPREL:
      return 0;
    default:
      return 0;
    }
  }

  uint32_t getPairRelocation(const Elf_Rel &rel) {
    switch (rel.getType(isMips64EL())) {
    case llvm::ELF::R_MIPS_HI16:
      return llvm::ELF::R_MIPS_LO16;
    case llvm::ELF::R_MIPS_GOT16: {
      const Elf_Sym *symbol =
          this->_objFile->getSymbol(rel.getSymbol(isMips64EL()));
      if (symbol->getBinding() == llvm::ELF::STB_LOCAL)
        return llvm::ELF::R_MIPS_LO16;
    }
    }
    return llvm::ELF::R_MIPS_NONE;
  }

  Elf_Rel_Iter findMatchingRelocation(uint32_t pairRelType, Elf_Rel_Iter rit,
                                      Elf_Rel_Iter eit) {
    return std::find_if(rit, eit, [&](const Elf_Rel &rel) {
      return rel.getType(isMips64EL()) == pairRelType &&
             rel.getSymbol(isMips64EL()) == rit->getSymbol(isMips64EL());
    });
  }

  bool isMips64EL() const { return this->_objFile->isMips64EL(); }
};

template <class ELFT> class MipsDynamicFile : public DynamicFile<ELFT> {
public:
  MipsDynamicFile(const MipsLinkingContext &context, StringRef name)
      : DynamicFile<ELFT>(context, name) {}
};

} // elf
} // lld

#endif
