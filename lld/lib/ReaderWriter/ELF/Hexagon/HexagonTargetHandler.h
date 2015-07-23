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

  const AtomLayout *appendAtom(const Atom *atom) override;
};

/// \brief TargetLayout for Hexagon
class HexagonTargetLayout final : public TargetLayout<ELF32LE> {
public:
  enum HexagonSectionOrder {
    ORDER_SDATA = 205
  };

  HexagonTargetLayout(HexagonLinkingContext &ctx)
      : TargetLayout(ctx), _sdataSection(ctx) {}

  /// \brief Return the section order for a input section
  TargetLayout::SectionOrder
  getSectionOrder(StringRef name, int32_t contentType,
                  int32_t contentPermissions) override {
    if (contentType == DefinedAtom::typeDataFast ||
        contentType == DefinedAtom::typeZeroFillFast)
      return ORDER_SDATA;
    return TargetLayout::getSectionOrder(name, contentType, contentPermissions);
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
    return TargetLayout::getInputSectionName(da);
  }

  /// \brief Gets or creates a section.
  AtomSection<ELF32LE> *
  createSection(StringRef name, int32_t contentType,
                DefinedAtom::ContentPermissions contentPermissions,
                TargetLayout::SectionOrder sectionOrder) override {
    if (contentType == DefinedAtom::typeDataFast ||
        contentType == DefinedAtom::typeZeroFillFast)
      return &_sdataSection;
    return TargetLayout::createSection(name, contentType, contentPermissions,
                                       sectionOrder);
  }

  /// \brief get the segment type for the section thats defined by the target
  TargetLayout::SegmentType
  getSegmentType(const Section<ELF32LE> *section) const override {
    if (section->order() == ORDER_SDATA)
      return PT_LOAD;
    return TargetLayout::getSegmentType(section);
  }

  Section<ELF32LE> *getSDataSection() { return &_sdataSection; }

  uint64_t getGOTSymAddr() {
    std::call_once(_gotOnce, [this]() {
      if (AtomLayout *got = findAbsoluteAtom("_GLOBAL_OFFSET_TABLE_"))
        _gotAddr = got->_virtualAddr;
    });
    return _gotAddr;
  }

private:
  SDataSection _sdataSection;
  uint64_t _gotAddr = 0;
  std::once_flag _gotOnce;
};

/// \brief TargetHandler for Hexagon
class HexagonTargetHandler final : public TargetHandler {
public:
  HexagonTargetHandler(HexagonLinkingContext &targetInfo);

  const TargetRelocationHandler &getRelocationHandler() const override {
    return *_relocationHandler;
  }

  std::unique_ptr<Reader> getObjReader() override {
    return llvm::make_unique<ELFReader<HexagonELFFile>>(_ctx);
  }

  std::unique_ptr<Reader> getDSOReader() override {
    return llvm::make_unique<ELFReader<DynamicFile<ELF32LE>>>(_ctx);
  }

  std::unique_ptr<Writer> getWriter() override;

private:
  HexagonLinkingContext &_ctx;
  std::unique_ptr<HexagonTargetLayout> _targetLayout;
  std::unique_ptr<HexagonTargetRelocationHandler> _relocationHandler;
};

void finalizeHexagonRuntimeAtomValues(HexagonTargetLayout &layout);

} // end namespace elf
} // end namespace lld

#endif
