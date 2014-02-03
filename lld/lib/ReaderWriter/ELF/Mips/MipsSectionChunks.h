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
  MipsGOTSection(const MipsLinkingContext &context)
      : AtomSection<ELFType>(context, ".got", DefinedAtom::typeGOT,
                             DefinedAtom::permRW_,
                             MipsTargetLayout<ELFType>::ORDER_GOT),
        _globalCount(0) {
    this->_flags |= SHF_MIPS_GPREL;
    this->_align2 = 4;
  }

  /// \brief Number of local GOT entries.
  std::size_t getLocalCount() const {
    return this->_atoms.size() - _globalCount;
  }

  /// \brief Number of global GOT entries.
  std::size_t getGlobalCount() const { return _globalCount; }

  /// \brief Compare two atoms accordingly theirs positions in the GOT.
  bool compare(const Atom *a, const Atom *b) const {
    auto ia = _posMap.find(a);
    auto ib = _posMap.find(b);

    if (ia != _posMap.end() && ib != _posMap.end())
      return ia->second < ib->second;

    return ia == _posMap.end() && ib != _posMap.end();
  }

  virtual const lld::AtomLayout &appendAtom(const Atom *atom) {
    const DefinedAtom *da = dyn_cast<DefinedAtom>(atom);

    const Atom *ta = nullptr;
    for (const auto &r : *da) {
      if (r->kindNamespace() != lld::Reference::KindNamespace::ELF)
        continue;
      assert(r->kindArch() == Reference::KindArch::Mips);
      if (r->kindValue() == LLD_R_MIPS_GLOBAL_GOT) {
        ta = r->target();
        break;
      }
    }

    if (ta) {
      _posMap[ta] = _posMap.size();
      ++_globalCount;
    }

    return AtomSection<ELFType>::appendAtom(atom);
  }

private:
  /// \brief Number of global GOT entries.
  std::size_t _globalCount;

  /// \brief Map Atoms to their GOT entry index.
  llvm::DenseMap<const Atom *, std::size_t> _posMap;
};

} // elf
} // lld

#endif
