//===- lib/ReaderWriter/ELF/Hexagon/HexagonSectionChunks.h-----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_SECTION_CHUNKS_H
#define HEXAGON_SECTION_CHUNKS_H

#include "HexagonTargetHandler.h"

namespace lld {
namespace elf {
template <typename ELFT> class HexagonTargetLayout;
class HexagonLinkingContext;

/// \brief Handle Hexagon SData section
template <class ELFT> class SDataSection : public AtomSection<ELFT> {
public:
  SDataSection(const HexagonLinkingContext &ctx)
      : AtomSection<ELFT>(ctx, ".sdata", DefinedAtom::typeDataFast, 0,
                          HexagonTargetLayout<ELFT>::ORDER_SDATA) {
    this->_type = SHT_PROGBITS;
    this->_flags = SHF_ALLOC | SHF_WRITE;
    this->_alignment = 4096;
  }

  /// \brief Finalize the section contents before writing
  void doPreFlight() override;

  /// \brief Does this section have an output segment.
  bool hasOutputSegment() const override { return true; }

  const lld::AtomLayout *appendAtom(const Atom *atom) override {
    const DefinedAtom *definedAtom = cast<DefinedAtom>(atom);
    DefinedAtom::Alignment atomAlign = definedAtom->alignment();
    uint64_t alignment = atomAlign.value;
    this->_atoms.push_back(new (this->_alloc) lld::AtomLayout(atom, 0, 0));
    // Set the section alignment to the largest alignment
    // std::max doesn't support uint64_t
    if (this->_alignment < alignment)
      this->_alignment = alignment;
    return (this->_atoms.back());
  }

}; // SDataSection

template <class ELFT> void SDataSection<ELFT>::doPreFlight() {
  // sort the atoms on the alignments they have been set
  std::stable_sort(this->_atoms.begin(), this->_atoms.end(),
                                             [](const lld::AtomLayout * A,
                                                const lld::AtomLayout * B) {
    const DefinedAtom *definedAtomA = cast<DefinedAtom>(A->_atom);
    const DefinedAtom *definedAtomB = cast<DefinedAtom>(B->_atom);
    int64_t alignmentA = definedAtomA->alignment().value;
    int64_t alignmentB = definedAtomB->alignment().value;
    if (alignmentA == alignmentB) {
      if (definedAtomA->merge() == DefinedAtom::mergeAsTentative)
        return false;
      if (definedAtomB->merge() == DefinedAtom::mergeAsTentative)
        return true;
    }
    return alignmentA < alignmentB;
  });

  // Set the fileOffset, and the appropriate size of the section
  for (auto &ai : this->_atoms) {
    const DefinedAtom *definedAtom = cast<DefinedAtom>(ai->_atom);
    DefinedAtom::Alignment atomAlign = definedAtom->alignment();
    uint64_t fOffset = this->alignOffset(this->fileSize(), atomAlign);
    uint64_t mOffset = this->alignOffset(this->memSize(), atomAlign);
    ai->_fileOffset = fOffset;
    this->_fsize = fOffset + definedAtom->size();
    this->_msize = mOffset + definedAtom->size();
  }
} // finalize

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_HEXAGON_HEXAGON_SECTION_CHUNKS_H
