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
#include "HexagonSectionChunks.h"
#include "TargetLayout.h"

namespace lld {
namespace elf {
class HexagonLinkingContext;

/// \brief TargetLayout for Hexagon
template <class HexagonELFType>
class HexagonTargetLayout final : public TargetLayout<HexagonELFType> {
public:
  enum HexagonSectionOrder {
    ORDER_SDATA = 205
  };

  HexagonTargetLayout(HexagonLinkingContext &hti)
      : TargetLayout<HexagonELFType>(hti), _sdataSection() {
    _sdataSection = new (_alloc) SDataSection<HexagonELFType>(hti);
  }

  /// \brief Return the section order for a input section
  typename TargetLayout<HexagonELFType>::SectionOrder
  getSectionOrder(StringRef name, int32_t contentType,
                  int32_t contentPermissions) override {
    if ((contentType == DefinedAtom::typeDataFast) ||
       (contentType == DefinedAtom::typeZeroFillFast))
      return ORDER_SDATA;

    return TargetLayout<HexagonELFType>::getSectionOrder(name, contentType,
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
    return TargetLayout<HexagonELFType>::getInputSectionName(da);
  }

  /// \brief Gets or creates a section.
  AtomSection<HexagonELFType> *
  createSection(StringRef name, int32_t contentType,
                DefinedAtom::ContentPermissions contentPermissions,
                typename TargetLayout<HexagonELFType>::SectionOrder
                    sectionOrder) override {
    if ((contentType == DefinedAtom::typeDataFast) ||
       (contentType == DefinedAtom::typeZeroFillFast))
      return _sdataSection;
    return TargetLayout<HexagonELFType>::createSection(
        name, contentType, contentPermissions, sectionOrder);
  }

  /// \brief get the segment type for the section thats defined by the target
  typename TargetLayout<HexagonELFType>::SegmentType
  getSegmentType(Section<HexagonELFType> *section) const override {
    if (section->order() == ORDER_SDATA)
      return PT_LOAD;

    return TargetLayout<HexagonELFType>::getSegmentType(section);
  }

  Section<HexagonELFType> *getSDataSection() const {
    return _sdataSection;
  }

  uint64_t getGOTSymAddr() {
    if (!_gotSymAtom.hasValue())
      _gotSymAtom = this->findAbsoluteAtom("_GLOBAL_OFFSET_TABLE_");
    if (*_gotSymAtom)
      return (*_gotSymAtom)->_virtualAddr;
    return 0;
  }

private:
  llvm::BumpPtrAllocator _alloc;
  SDataSection<HexagonELFType> *_sdataSection = nullptr;
  llvm::Optional<AtomLayout *> _gotSymAtom;
};

/// \brief TargetHandler for Hexagon
class HexagonTargetHandler final : public TargetHandler {
  typedef llvm::object::ELFType<llvm::support::little, 2, false> ELFTy;
  typedef ELFObjectReader<ELFTy, HexagonLinkingContext,
                          lld::elf::HexagonELFFile> ObjReader;
  typedef ELFDSOReader<ELFTy, HexagonLinkingContext> ELFDSOReader;

public:
  HexagonTargetHandler(HexagonLinkingContext &targetInfo);

  const HexagonTargetRelocationHandler &getRelocationHandler() const override {
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
  std::unique_ptr<HexagonRuntimeFile<ELFTy>> _runtimeFile;
  std::unique_ptr<HexagonTargetLayout<ELFTy>> _targetLayout;
  std::unique_ptr<HexagonTargetRelocationHandler> _relocationHandler;
};

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
