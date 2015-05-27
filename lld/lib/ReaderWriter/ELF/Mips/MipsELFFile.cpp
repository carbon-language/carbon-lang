//===- lib/ReaderWriter/ELF/MipsELFFile.cpp -------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MipsELFFile.h"

namespace lld {
namespace elf {

template <class ELFT>
MipsELFDefinedAtom<ELFT>::MipsELFDefinedAtom(
    const MipsELFFile<ELFT> &file, StringRef symbolName, StringRef sectionName,
    const Elf_Sym *symbol, const Elf_Shdr *section,
    ArrayRef<uint8_t> contentData, unsigned int referenceStart,
    unsigned int referenceEnd, std::vector<ELFReference<ELFT> *> &referenceList)
    : ELFDefinedAtom<ELFT>(file, symbolName, sectionName, symbol, section,
                           contentData, referenceStart, referenceEnd,
                           referenceList) {}

template <class ELFT>
const MipsELFFile<ELFT> &MipsELFDefinedAtom<ELFT>::file() const {
  return static_cast<const MipsELFFile<ELFT> &>(this->_owningFile);
}

template <class ELFT>
DefinedAtom::CodeModel MipsELFDefinedAtom<ELFT>::codeModel() const {
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

template class MipsELFDefinedAtom<ELF32LE>;
template class MipsELFDefinedAtom<ELF64LE>;

template <class ELFT> static bool isMips64EL() {
  return ELFT::Is64Bits && ELFT::TargetEndianness == llvm::support::little;
}

template <class ELFT, bool isRela>
static uint32_t
extractTag(const llvm::object::Elf_Rel_Impl<ELFT, isRela> &rel) {
  return (rel.getType(isMips64EL<ELFT>()) & 0xffffff00) >> 8;
}

template <class ELFT>
MipsELFReference<ELFT>::MipsELFReference(uint64_t symValue, const Elf_Rela &rel)
    : ELFReference<ELFT>(&rel, rel.r_offset - symValue,
                         Reference::KindArch::Mips,
                         rel.getType(isMips64EL<ELFT>()) & 0xff,
                         rel.getSymbol(isMips64EL<ELFT>())),
      _tag(extractTag(rel)) {}

template <class ELFT>
MipsELFReference<ELFT>::MipsELFReference(uint64_t symValue, const Elf_Rel &rel)
    : ELFReference<ELFT>(rel.r_offset - symValue, Reference::KindArch::Mips,
                         rel.getType(isMips64EL<ELFT>()) & 0xff,
                         rel.getSymbol(isMips64EL<ELFT>())),
      _tag(extractTag(rel)) {}

template class MipsELFReference<ELF32LE>;
template class MipsELFReference<ELF64LE>;

template <class ELFT>
MipsELFFile<ELFT>::MipsELFFile(std::unique_ptr<MemoryBuffer> mb,
                               ELFLinkingContext &ctx)
    : ELFFile<ELFT>(std::move(mb), ctx) {}

template <class ELFT> bool MipsELFFile<ELFT>::isPIC() const {
  return this->_objFile->getHeader()->e_flags & llvm::ELF::EF_MIPS_PIC;
}

template <class ELFT> std::error_code MipsELFFile<ELFT>::doParse() {
  if (std::error_code ec = ELFFile<ELFT>::doParse())
    return ec;
  // Retrieve some auxiliary data like GP value, TLS section address etc
  // from the object file.
  return readAuxData();
}

template <class ELFT>
ELFDefinedAtom<ELFT> *MipsELFFile<ELFT>::createDefinedAtom(
    StringRef symName, StringRef sectionName, const Elf_Sym *sym,
    const Elf_Shdr *sectionHdr, ArrayRef<uint8_t> contentData,
    unsigned int referenceStart, unsigned int referenceEnd,
    std::vector<ELFReference<ELFT> *> &referenceList) {
  return new (this->_readerStorage) MipsELFDefinedAtom<ELFT>(
      *this, symName, sectionName, sym, sectionHdr, contentData, referenceStart,
      referenceEnd, referenceList);
}

template <class ELFT>
const typename MipsELFFile<ELFT>::Elf_Shdr *
MipsELFFile<ELFT>::findSectionByType(uint64_t type) const {
  for (const Elf_Shdr &section : this->_objFile->sections())
    if (section.sh_type == type)
      return &section;
  return nullptr;
}

template <class ELFT>
const typename MipsELFFile<ELFT>::Elf_Shdr *
MipsELFFile<ELFT>::findSectionByFlags(uint64_t flags) const {
  for (const Elf_Shdr &section : this->_objFile->sections())
    if (section.sh_flags & flags)
      return &section;
  return nullptr;
}

template <class ELFT>
ErrorOr<const typename MipsELFFile<ELFT>::Elf_Mips_RegInfo *>
MipsELFFile<ELFT>::findRegInfoSec() const {
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

      const auto *opt = reinterpret_cast<const Elf_Mips_Options *>(raw.data());
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

template <class ELFT> std::error_code MipsELFFile<ELFT>::readAuxData() {
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

template <class ELFT>
void MipsELFFile<ELFT>::createRelocationReferences(const Elf_Sym *symbol,
                                                   ArrayRef<uint8_t> content,
                                                   range<Elf_Rela_Iter> rels) {
  const auto value = this->getSymbolValue(symbol);
  for (const auto &rel : rels) {
    if (rel.r_offset < value || value + content.size() <= rel.r_offset)
      continue;
    auto r = new (this->_readerStorage) MipsELFReference<ELFT>(value, rel);
    this->addReferenceToSymbol(r, symbol);
    this->_references.push_back(r);
  }
}

template <class ELFT>
void MipsELFFile<ELFT>::createRelocationReferences(const Elf_Sym *symbol,
                                                   ArrayRef<uint8_t> symContent,
                                                   ArrayRef<uint8_t> secContent,
                                                   range<Elf_Rel_Iter> rels) {
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

template <class ELFT>
static uint8_t
getPrimaryType(const llvm::object::Elf_Rel_Impl<ELFT, false> &rel) {
  return rel.getType(isMips64EL<ELFT>()) & 0xff;
}

template <class ELFT>
Reference::Addend
MipsELFFile<ELFT>::readAddend(const Elf_Rel &ri,
                              const ArrayRef<uint8_t> content) const {
  return readMipsRelocAddend(getPrimaryType(ri), content.data() + ri.r_offset);
}

template <class ELFT>
uint32_t MipsELFFile<ELFT>::getPairRelocation(const Elf_Rel &rel) const {
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

template <class ELFT>
typename MipsELFFile<ELFT>::Elf_Rel_Iter
MipsELFFile<ELFT>::findMatchingRelocation(uint32_t pairRelType,
                                          Elf_Rel_Iter rit,
                                          Elf_Rel_Iter eit) const {
  return std::find_if(rit, eit, [&](const Elf_Rel &rel) {
    return getPrimaryType(rel) == pairRelType &&
           rel.getSymbol(isMips64EL<ELFT>()) ==
               rit->getSymbol(isMips64EL<ELFT>());
  });
}

template <class ELFT>
bool MipsELFFile<ELFT>::isLocalBinding(const Elf_Rel &rel) const {
  return this->_objFile->getSymbol(rel.getSymbol(isMips64EL<ELFT>()))
             ->getBinding() == llvm::ELF::STB_LOCAL;
}

template class MipsELFFile<ELF32LE>;
template class MipsELFFile<ELF64LE>;

} // elf
} // lld
