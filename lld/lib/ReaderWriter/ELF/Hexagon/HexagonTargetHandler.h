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
#include "HexagonRelocationHandler.h"
#include "TargetLayout.h"

namespace lld {
namespace elf {
class HexagonLinkingContext;

/// \brief Handle Hexagon SData section
class SDataSection : public AtomSection<ELF32LE> {
public:
  SDataSection(const HexagonLinkingContext &ctx);

  /// \brief Finalize the section contents before writing
  void doPreFlight() override;

  /// \brief Does this section have an output segment.
  bool hasOutputSegment() const override { return true; }

  const lld::AtomLayout *appendAtom(const Atom *atom) override;
};

/// \brief TargetLayout for Hexagon
class HexagonTargetLayout final : public TargetLayout<ELF32LE> {
public:
  enum HexagonSectionOrder {
    ORDER_SDATA = 205
  };

  HexagonTargetLayout(HexagonLinkingContext &hti)
      : TargetLayout<ELF32LE>(hti), _sdataSection() {
    _sdataSection = new (_alloc) SDataSection(hti);
  }

  /// \brief Return the section order for a input section
  TargetLayout<ELF32LE>::SectionOrder
  getSectionOrder(StringRef name, int32_t contentType,
                  int32_t contentPermissions) override {
    if ((contentType == DefinedAtom::typeDataFast) ||
       (contentType == DefinedAtom::typeZeroFillFast))
      return ORDER_SDATA;

    return TargetLayout<ELF32LE>::getSectionOrder(name, contentType,
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
    return TargetLayout<ELF32LE>::getInputSectionName(da);
  }

  /// \brief Gets or creates a section.
  AtomSection<ELF32LE> *
  createSection(StringRef name, int32_t contentType,
                DefinedAtom::ContentPermissions contentPermissions,
                TargetLayout<ELF32LE>::SectionOrder sectionOrder) override {
    if ((contentType == DefinedAtom::typeDataFast) ||
       (contentType == DefinedAtom::typeZeroFillFast))
      return _sdataSection;
    return TargetLayout<ELF32LE>::createSection(
        name, contentType, contentPermissions, sectionOrder);
  }

  /// \brief get the segment type for the section thats defined by the target
  TargetLayout<ELF32LE>::SegmentType
  getSegmentType(Section<ELF32LE> *section) const override {
    if (section->order() == ORDER_SDATA)
      return PT_LOAD;

    return TargetLayout<ELF32LE>::getSegmentType(section);
  }

  Section<ELF32LE> *getSDataSection() const { return _sdataSection; }

  uint64_t getGOTSymAddr() {
    std::call_once(_gotOnce, [this]() {
      if (AtomLayout *got = findAbsoluteAtom("_GLOBAL_OFFSET_TABLE_"))
        _gotAddr = got->_virtualAddr;
    });
    return _gotAddr;
  }

private:
  llvm::BumpPtrAllocator _alloc;
  SDataSection *_sdataSection = nullptr;
  uint64_t _gotAddr = 0;
  std::once_flag _gotOnce;
};

/// \brief TargetHandler for Hexagon
class HexagonTargetHandler final : public TargetHandler {
  typedef ELFReader<HexagonELFFile> ObjReader;
  typedef ELFReader<DynamicFile<ELF32LE>> ELFDSOReader;

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
  std::unique_ptr<HexagonTargetLayout> _targetLayout;
  std::unique_ptr<HexagonTargetRelocationHandler> _relocationHandler;
};

inline void SDataSection::doPreFlight() {
  // sort the atoms on the alignments they have been set
  std::stable_sort(_atoms.begin(), _atoms.end(), [](const lld::AtomLayout *A,
                                                    const lld::AtomLayout *B) {
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
  for (auto &ai : _atoms) {
    const DefinedAtom *definedAtom = cast<DefinedAtom>(ai->_atom);
    DefinedAtom::Alignment atomAlign = definedAtom->alignment();
    uint64_t fOffset = alignOffset(fileSize(), atomAlign);
    uint64_t mOffset = alignOffset(memSize(), atomAlign);
    ai->_fileOffset = fOffset;
    _fsize = fOffset + definedAtom->size();
    _msize = mOffset + definedAtom->size();
  }
} // finalize

inline SDataSection::SDataSection(const HexagonLinkingContext &ctx)
    : AtomSection<ELF32LE>(ctx, ".sdata", DefinedAtom::typeDataFast, 0,
                           HexagonTargetLayout::ORDER_SDATA) {
  _type = SHT_PROGBITS;
  _flags = SHF_ALLOC | SHF_WRITE;
  _alignment = 4096;
}

inline const lld::AtomLayout *SDataSection::appendAtom(const Atom *atom) {
  const DefinedAtom *definedAtom = cast<DefinedAtom>(atom);
  DefinedAtom::Alignment atomAlign = definedAtom->alignment();
  uint64_t alignment = atomAlign.value;
  _atoms.push_back(new (_alloc) lld::AtomLayout(atom, 0, 0));
  // Set the section alignment to the largest alignment
  // std::max doesn't support uint64_t
  if (_alignment < alignment)
    _alignment = alignment;
  return _atoms.back();
}

inline void finalizeHexagonRuntimeAtomValues(HexagonTargetLayout &layout) {
  AtomLayout *gotAtom = layout.findAbsoluteAtom("_GLOBAL_OFFSET_TABLE_");
  OutputSection<ELF32LE> *gotpltSection = layout.findOutputSection(".got.plt");
  if (gotpltSection)
    gotAtom->_virtualAddr = gotpltSection->virtualAddr();
  else
    gotAtom->_virtualAddr = 0;
  AtomLayout *dynamicAtom = layout.findAbsoluteAtom("_DYNAMIC");
  OutputSection<ELF32LE> *dynamicSection = layout.findOutputSection(".dynamic");
  if (dynamicSection)
    dynamicAtom->_virtualAddr = dynamicSection->virtualAddr();
  else
    dynamicAtom->_virtualAddr = 0;
}

} // end namespace elf
} // end namespace lld

#endif
