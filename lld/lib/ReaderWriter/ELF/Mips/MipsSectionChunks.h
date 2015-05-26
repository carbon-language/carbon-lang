//===- lib/ReaderWriter/ELF/Mips/MipsSectionChunks.h ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_MIPS_MIPS_SECTION_CHUNKS_H
#define LLD_READER_WRITER_ELF_MIPS_MIPS_SECTION_CHUNKS_H

#include "MipsReginfo.h"
#include "SectionChunks.h"

namespace lld {
namespace elf {

template <typename ELFT> class MipsTargetLayout;
class MipsLinkingContext;

/// \brief Handle Mips .reginfo section
template <class ELFT> class MipsReginfoSection : public Section<ELFT> {
public:
  MipsReginfoSection(const ELFLinkingContext &ctx,
                     MipsTargetLayout<ELFT> &targetLayout,
                     const MipsReginfo &reginfo)
      : Section<ELFT>(ctx, ".reginfo", "MipsReginfo"),
        _targetLayout(targetLayout) {
    this->setOrder(MipsTargetLayout<ELFT>::ORDER_MIPS_REGINFO);
    this->_entSize = sizeof(Elf_Mips_RegInfo);
    this->_fsize = sizeof(Elf_Mips_RegInfo);
    this->_msize = sizeof(Elf_Mips_RegInfo);
    this->_alignment = 4;
    this->_type = SHT_MIPS_REGINFO;
    this->_flags = SHF_ALLOC;

    std::memset(&_reginfo, 0, sizeof(_reginfo));
    _reginfo.ri_gprmask = reginfo._gpRegMask;
    _reginfo.ri_cprmask[0] = reginfo._cpRegMask[0];
    _reginfo.ri_cprmask[1] = reginfo._cpRegMask[1];
    _reginfo.ri_cprmask[2] = reginfo._cpRegMask[2];
    _reginfo.ri_cprmask[3] = reginfo._cpRegMask[3];
  }

  StringRef segmentKindToStr() const override { return "REGINFO"; }

  bool hasOutputSegment() const override { return true; }

  void write(ELFWriter *writer, TargetLayout<ELFT> &layout,
             llvm::FileOutputBuffer &buffer) override {
    uint8_t *dest = buffer.getBufferStart() + this->fileOffset();
    std::memcpy(dest, &_reginfo, this->_fsize);
  }

  void finalize() override {
    _reginfo.ri_gp_value = _targetLayout.getGPAddr();

    if (this->_outputSection)
      this->_outputSection->setType(this->_type);
  }

private:
  typedef llvm::object::Elf_Mips_RegInfo<ELFT> Elf_Mips_RegInfo;

  Elf_Mips_RegInfo _reginfo;
  MipsTargetLayout<ELFT> &_targetLayout;
};

/// \brief Handle .MIPS.options section
template <class ELFT> class MipsOptionsSection : public Section<ELFT> {
public:
  typedef typename std::vector<MipsReginfo>::const_iterator mask_const_iterator;

  MipsOptionsSection(const ELFLinkingContext &ctx,
                     MipsTargetLayout<ELFT> &targetLayout,
                     const MipsReginfo &reginfo)
      : Section<ELFT>(ctx, ".MIPS.options", "MipsOptions"),
        _targetLayout(targetLayout) {
    this->setOrder(MipsTargetLayout<ELFT>::ORDER_MIPS_OPTIONS);
    this->_entSize = 1;
    this->_alignment = 8;
    this->_fsize = llvm::RoundUpToAlignment(
        sizeof(Elf_Mips_Options) + sizeof(Elf_Mips_RegInfo), this->_alignment);
    this->_msize = this->_fsize;
    this->_type = SHT_MIPS_OPTIONS;
    this->_flags = SHF_ALLOC | SHF_MIPS_NOSTRIP;

    _header.kind = ODK_REGINFO;
    _header.size = this->_fsize;
    _header.section = 0;
    _header.info = 0;

    std::memset(&_reginfo, 0, sizeof(_reginfo));
    _reginfo.ri_gprmask = reginfo._gpRegMask;
    _reginfo.ri_cprmask[0] = reginfo._cpRegMask[0];
    _reginfo.ri_cprmask[1] = reginfo._cpRegMask[1];
    _reginfo.ri_cprmask[2] = reginfo._cpRegMask[2];
    _reginfo.ri_cprmask[3] = reginfo._cpRegMask[3];
  }

  bool hasOutputSegment() const override { return true; }

  void write(ELFWriter *writer, TargetLayout<ELFT> &layout,
             llvm::FileOutputBuffer &buffer) override {
    uint8_t *dest = buffer.getBufferStart() + this->fileOffset();
    std::memset(dest, 0, this->_fsize);
    std::memcpy(dest, &_header, sizeof(_header));
    std::memcpy(dest + sizeof(_header), &_reginfo, sizeof(_reginfo));
  }

  void finalize() override {
    _reginfo.ri_gp_value = _targetLayout.getGPAddr();

    if (this->_outputSection)
      this->_outputSection->setType(this->_type);
  }

private:
  typedef llvm::object::Elf_Mips_Options<ELFT> Elf_Mips_Options;
  typedef llvm::object::Elf_Mips_RegInfo<ELFT> Elf_Mips_RegInfo;

  Elf_Mips_Options _header;
  Elf_Mips_RegInfo _reginfo;
  MipsTargetLayout<ELFT> &_targetLayout;
};

/// \brief Handle Mips GOT section
template <class ELFT> class MipsGOTSection : public AtomSection<ELFT> {
public:
  MipsGOTSection(const MipsLinkingContext &ctx)
      : AtomSection<ELFT>(ctx, ".got", DefinedAtom::typeGOT,
                          DefinedAtom::permRW_,
                          MipsTargetLayout<ELFT>::ORDER_GOT),
        _hasNonLocal(false), _localCount(0) {
    this->_flags |= SHF_MIPS_GPREL;
    this->_alignment = 4;
  }

  /// \brief Number of local GOT entries.
  std::size_t getLocalCount() const { return _localCount; }

  /// \brief Number of global GOT entries.
  std::size_t getGlobalCount() const { return _posMap.size(); }

  /// \brief Does the atom have a global GOT entry?
  bool hasGlobalGOTEntry(const Atom *a) const {
    return _posMap.count(a) || _tlsMap.count(a);
  }

  /// \brief Compare two atoms accordingly theirs positions in the GOT.
  bool compare(const Atom *a, const Atom *b) const {
    auto ia = _posMap.find(a);
    auto ib = _posMap.find(b);

    if (ia != _posMap.end() && ib != _posMap.end())
      return ia->second < ib->second;

    return ia == _posMap.end() && ib != _posMap.end();
  }

  const AtomLayout *appendAtom(const Atom *atom) override {
    const DefinedAtom *da = dyn_cast<DefinedAtom>(atom);

    if (atom->name() == "_GLOBAL_OFFSET_TABLE_")
      return AtomSection<ELFT>::appendAtom(atom);

    for (const auto &r : *da) {
      if (r->kindNamespace() != Reference::KindNamespace::ELF)
        continue;
      assert(r->kindArch() == Reference::KindArch::Mips);
      switch (r->kindValue()) {
      case LLD_R_MIPS_GLOBAL_GOT:
        _hasNonLocal = true;
        _posMap[r->target()] = _posMap.size();
        return AtomSection<ELFT>::appendAtom(atom);
      case R_MIPS_TLS_TPREL32:
      case R_MIPS_TLS_DTPREL32:
      case R_MIPS_TLS_TPREL64:
      case R_MIPS_TLS_DTPREL64:
        _hasNonLocal = true;
        _tlsMap[r->target()] = _tlsMap.size();
        return AtomSection<ELFT>::appendAtom(atom);
      case R_MIPS_TLS_DTPMOD32:
      case R_MIPS_TLS_DTPMOD64:
        _hasNonLocal = true;
        break;
      }
    }

    if (!_hasNonLocal)
      ++_localCount;

    return AtomSection<ELFT>::appendAtom(atom);
  }

private:
  /// \brief True if the GOT contains non-local entries.
  bool _hasNonLocal;

  /// \brief Number of local GOT entries.
  std::size_t _localCount;

  /// \brief Map TLS Atoms to their GOT entry index.
  llvm::DenseMap<const Atom *, std::size_t> _tlsMap;

  /// \brief Map Atoms to their GOT entry index.
  llvm::DenseMap<const Atom *, std::size_t> _posMap;
};

/// \brief Handle Mips PLT section
template <class ELFT> class MipsPLTSection : public AtomSection<ELFT> {
public:
  MipsPLTSection(const MipsLinkingContext &ctx)
      : AtomSection<ELFT>(ctx, ".plt", DefinedAtom::typeGOT,
                          DefinedAtom::permR_X,
                          MipsTargetLayout<ELFT>::ORDER_PLT) {}

  const AtomLayout *findPLTLayout(const Atom *plt) const {
    auto it = _pltLayoutMap.find(plt);
    return it != _pltLayoutMap.end() ? it->second : nullptr;
  }

  const AtomLayout *appendAtom(const Atom *atom) override {
    const auto *layout = AtomSection<ELFT>::appendAtom(atom);

    const DefinedAtom *da = cast<DefinedAtom>(atom);

    for (const auto &r : *da) {
      if (r->kindNamespace() != Reference::KindNamespace::ELF)
        continue;
      assert(r->kindArch() == Reference::KindArch::Mips);
      if (r->kindValue() == LLD_R_MIPS_STO_PLT) {
        _pltLayoutMap[r->target()] = layout;
        break;
      }
    }

    return layout;
  }

private:
  /// \brief Map PLT Atoms to their layouts.
  std::unordered_map<const Atom *, const AtomLayout *> _pltLayoutMap;
};

template <class ELFT> class MipsRelocationTable : public RelocationTable<ELFT> {
  typedef llvm::object::Elf_Rel_Impl<ELFT, false> Elf_Rel;
  typedef llvm::object::Elf_Rel_Impl<ELFT, true> Elf_Rela;

  static const bool _isMips64EL =
      ELFT::Is64Bits && ELFT::TargetEndianness == llvm::support::little;

public:
  MipsRelocationTable(const ELFLinkingContext &ctx, StringRef str,
                      int32_t order)
      : RelocationTable<ELFT>(ctx, str, order) {}

protected:
  void writeRela(ELFWriter *writer, Elf_Rela &r, const DefinedAtom &atom,
                 const Reference &ref) override {
    uint32_t rType = ref.kindValue() | (ref.tag() << 8);
    r.setSymbolAndType(this->getSymbolIndex(ref.target()), rType, _isMips64EL);
    r.r_offset = writer->addressOfAtom(&atom) + ref.offsetInAtom();
    // The addend is used only by relative relocations
    if (this->_ctx.isRelativeReloc(ref))
      r.r_addend = writer->addressOfAtom(ref.target()) + ref.addend();
    else
      r.r_addend = 0;
  }

  void writeRel(ELFWriter *writer, Elf_Rel &r, const DefinedAtom &atom,
                const Reference &ref) override {
    uint32_t rType = ref.kindValue() | (ref.tag() << 8);
    r.setSymbolAndType(this->getSymbolIndex(ref.target()), rType, _isMips64EL);
    r.r_offset = writer->addressOfAtom(&atom) + ref.offsetInAtom();
  }
};

} // elf
} // lld

#endif
