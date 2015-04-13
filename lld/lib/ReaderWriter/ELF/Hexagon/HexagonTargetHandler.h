//===- lib/ReaderWriter/ELF/Hexagon/HexagonTargetHandler.h ----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_TARGET_HANDLER_H
#define HEXAGON_TARGET_HANDLER_H

#include "ELFReader.h"
#include "HexagonELFFile.h"
#include "HexagonExecutableAtoms.h"
#include "HexagonRelocationHandler.h"
#include "TargetLayout.h"

namespace lld {
namespace elf {
class HexagonLinkingContext;

typedef llvm::object::ELFType<llvm::support::little, 2, false> ELFT;

/// \brief Handle Hexagon SData section
template <class ELFT> class SDataSection : public AtomSection<ELFT> {
public:
  SDataSection(const HexagonLinkingContext &ctx);

  /// \brief Finalize the section contents before writing
  void doPreFlight() override;

  /// \brief Does this section have an output segment.
  bool hasOutputSegment() const override { return true; }

  const lld::AtomLayout *appendAtom(const Atom *atom) override;
};

/// \brief TargetLayout for Hexagon
template <class ELFT>
class HexagonTargetLayout final : public TargetLayout<ELFT> {
public:
  enum HexagonSectionOrder {
    ORDER_SDATA = 205
  };

  HexagonTargetLayout(HexagonLinkingContext &hti)
      : TargetLayout<ELFT>(hti), _sdataSection() {
    _sdataSection = new (_alloc) SDataSection<ELFT>(hti);
  }

  /// \brief Return the section order for a input section
  typename TargetLayout<ELFT>::SectionOrder
  getSectionOrder(StringRef name, int32_t contentType,
                  int32_t contentPermissions) override {
    if ((contentType == DefinedAtom::typeDataFast) ||
       (contentType == DefinedAtom::typeZeroFillFast))
      return ORDER_SDATA;

    return TargetLayout<ELFT>::getSectionOrder(name, contentType,
                                               contentPermissions);
  }

  /// \brief Return the appropriate input section name.
  StringRef getInputSectionName(const DefinedAtom *da) const override {
    switch (da->contentType()) {
    case DefinedAtom::typeDataFast:
    case DefinedAtom::typeZeroFillFast:
      return ".sdata";
    default:
      break;
    }
    return TargetLayout<ELFT>::getInputSectionName(da);
  }

  /// \brief Gets or creates a section.
  AtomSection<ELFT> *createSection(
      StringRef name, int32_t contentType,
      DefinedAtom::ContentPermissions contentPermissions,
      typename TargetLayout<ELFT>::SectionOrder sectionOrder) override {
    if ((contentType == DefinedAtom::typeDataFast) ||
       (contentType == DefinedAtom::typeZeroFillFast))
      return _sdataSection;
    return TargetLayout<ELFT>::createSection(name, contentType,
                                             contentPermissions, sectionOrder);
  }

  /// \brief get the segment type for the section thats defined by the target
  typename TargetLayout<ELFT>::SegmentType
  getSegmentType(Section<ELFT> *section) const override {
    if (section->order() == ORDER_SDATA)
      return PT_LOAD;

    return TargetLayout<ELFT>::getSegmentType(section);
  }

  Section<ELFT> *getSDataSection() const { return _sdataSection; }

  uint64_t getGOTSymAddr() {
    std::call_once(_gotOnce, [this]() {
      if (AtomLayout *got = this->findAbsoluteAtom("_GLOBAL_OFFSET_TABLE_"))
        _gotAddr = got->_virtualAddr;
    });
    return _gotAddr;
  }

private:
  llvm::BumpPtrAllocator _alloc;
  SDataSection<ELFT> *_sdataSection = nullptr;
  uint64_t _gotAddr = 0;
  std::once_flag _gotOnce;
};

/// \brief TargetHandler for Hexagon
class HexagonTargetHandler final : public TargetHandler {
  typedef llvm::object::ELFType<llvm::support::little, 2, false> ELFT;
  typedef ELFReader<ELFT, HexagonLinkingContext, HexagonELFFile> ObjReader;
  typedef ELFReader<ELFT, HexagonLinkingContext, DynamicFile> ELFDSOReader;

public:
  HexagonTargetHandler(HexagonLinkingContext &targetInfo);

  const TargetRelocationHandler &getRelocationHandler() const override {
    return *_relocationHandler;
  }

  std::unique_ptr<Reader> getObjReader() override {
    return llvm::make_unique<ObjReader>(_ctx);
  }

  std::unique_ptr<Reader> getDSOReader() override {
    return llvm::make_unique<ELFDSOReader>(_ctx);
  }

  std::unique_ptr<Writer> getWriter() override;

private:
  HexagonLinkingContext &_ctx;
  std::unique_ptr<HexagonRuntimeFile<ELFT>> _runtimeFile;
  std::unique_ptr<HexagonTargetLayout<ELFT>> _targetLayout;
  std::unique_ptr<HexagonTargetRelocationHandler> _relocationHandler;
};

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

template <class ELFT>
SDataSection<ELFT>::SDataSection(const HexagonLinkingContext &ctx)
    : AtomSection<ELFT>(ctx, ".sdata", DefinedAtom::typeDataFast, 0,
                        HexagonTargetLayout<ELFT>::ORDER_SDATA) {
  this->_type = SHT_PROGBITS;
  this->_flags = SHF_ALLOC | SHF_WRITE;
  this->_alignment = 4096;
}

template <class ELFT>
const lld::AtomLayout *SDataSection<ELFT>::appendAtom(const Atom *atom) {
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

template <class ELFT>
void finalizeHexagonRuntimeAtomValues(HexagonTargetLayout<ELFT> &layout) {
  AtomLayout *gotAtom = layout.findAbsoluteAtom("_GLOBAL_OFFSET_TABLE_");
  OutputSection<ELFT> *gotpltSection = layout.findOutputSection(".got.plt");
  if (gotpltSection)
    gotAtom->_virtualAddr = gotpltSection->virtualAddr();
  else
    gotAtom->_virtualAddr = 0;
  AtomLayout *dynamicAtom = layout.findAbsoluteAtom("_DYNAMIC");
  OutputSection<ELFT> *dynamicSection = layout.findOutputSection(".dynamic");
  if (dynamicSection)
    dynamicAtom->_virtualAddr = dynamicSection->virtualAddr();
  else
    dynamicAtom->_virtualAddr = 0;
}

} // end namespace elf
} // end namespace lld

#endif
