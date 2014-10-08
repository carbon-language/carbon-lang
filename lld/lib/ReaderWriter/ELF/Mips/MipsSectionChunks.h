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

namespace lld {
namespace elf {

template <typename ELFT> class MipsTargetLayout;
class MipsLinkingContext;

/// \brief Handle Mips GOT section
template <class ELFType> class MipsGOTSection : public AtomSection<ELFType> {
public:
  MipsGOTSection(const MipsLinkingContext &ctx)
      : AtomSection<ELFType>(ctx, ".got", DefinedAtom::typeGOT,
                             DefinedAtom::permRW_,
                             MipsTargetLayout<ELFType>::ORDER_GOT),
        _hasNonLocal(false), _localCount(0) {
    this->_flags |= SHF_MIPS_GPREL;
    this->_align2 = 4;
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

  const lld::AtomLayout &appendAtom(const Atom *atom) override {
    const DefinedAtom *da = dyn_cast<DefinedAtom>(atom);

    for (const auto &r : *da) {
      if (r->kindNamespace() != lld::Reference::KindNamespace::ELF)
        continue;
      assert(r->kindArch() == Reference::KindArch::Mips);
      switch (r->kindValue()) {
      case LLD_R_MIPS_GLOBAL_GOT:
        _hasNonLocal = true;
        _posMap[r->target()] = _posMap.size();
        return AtomSection<ELFType>::appendAtom(atom);
      case R_MIPS_TLS_TPREL32:
      case R_MIPS_TLS_DTPREL32:
        _hasNonLocal = true;
        _tlsMap[r->target()] = _tlsMap.size();
        return AtomSection<ELFType>::appendAtom(atom);
      case R_MIPS_TLS_DTPMOD32:
        _hasNonLocal = true;
        break;
      }
    }

    if (!_hasNonLocal)
      ++_localCount;

    return AtomSection<ELFType>::appendAtom(atom);
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
template <class ELFType> class MipsPLTSection : public AtomSection<ELFType> {
public:
  MipsPLTSection(const MipsLinkingContext &ctx)
      : AtomSection<ELFType>(ctx, ".plt", DefinedAtom::typeGOT,
                             DefinedAtom::permR_X,
                             MipsTargetLayout<ELFType>::ORDER_PLT) {}

  const AtomLayout *findPLTLayout(const Atom *plt) const {
    auto it = _pltLayoutMap.find(plt);
    return it != _pltLayoutMap.end() ? it->second : nullptr;
  }

  const lld::AtomLayout &appendAtom(const Atom *atom) override {
    const auto &layout = AtomSection<ELFType>::appendAtom(atom);

    const DefinedAtom *da = cast<DefinedAtom>(atom);

    for (const auto &r : *da) {
      if (r->kindNamespace() != lld::Reference::KindNamespace::ELF)
        continue;
      assert(r->kindArch() == Reference::KindArch::Mips);
      if (r->kindValue() == LLD_R_MIPS_STO_PLT) {
        _pltLayoutMap[r->target()] = &layout;
        break;
      }
    }

    return layout;
  }

private:
  /// \brief Map PLT Atoms to their layouts.
  std::unordered_map<const Atom *, const AtomLayout *> _pltLayoutMap;
};

} // elf
} // lld

#endif
