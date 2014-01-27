//===- lib/ReaderWriter/ELF/Hexagon/HexagonTargetHandler.h ----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_HEXAGON_HEXAGON_TARGET_HANDLER_H
#define LLD_READER_WRITER_ELF_HEXAGON_HEXAGON_TARGET_HANDLER_H

#include "DefaultTargetHandler.h"
#include "HexagonExecutableAtoms.h"
#include "HexagonRelocationHandler.h"
#include "HexagonSectionChunks.h"
#include "TargetLayout.h"
#include "HexagonELFReader.h"

namespace lld {
namespace elf {
typedef llvm::object::ELFType<llvm::support::little, 2, false> HexagonELFType;
class HexagonLinkingContext;

/// \brief TargetLayout for Hexagon
template <class HexagonELFType>
class HexagonTargetLayout LLVM_FINAL : public TargetLayout<HexagonELFType> {

public:
  enum HexagonSectionOrder {
    ORDER_SDATA = 205
  };

  HexagonTargetLayout(const HexagonLinkingContext &hti)
      : TargetLayout<HexagonELFType>(hti), _sdataSection(nullptr) {
    _sdataSection = new (_alloc) SDataSection<HexagonELFType>(hti);
  }

  /// \brief Return the section order for a input section
  virtual Layout::SectionOrder getSectionOrder(
      StringRef name, int32_t contentType, int32_t contentPermissions) {
    if ((contentType == DefinedAtom::typeDataFast) ||
       (contentType == DefinedAtom::typeZeroFillFast))
      return ORDER_SDATA;

    return DefaultLayout<HexagonELFType>::getSectionOrder(name, contentType,
                                                          contentPermissions);
  }

  /// \brief This maps the input sections to the output section names
  virtual StringRef getSectionName(const DefinedAtom *da) const {
    switch (da->contentType()) {
    case DefinedAtom::typeDataFast:
    case DefinedAtom::typeZeroFillFast:
      return ".sdata";
    default:
      break;
    }
    return DefaultLayout<HexagonELFType>::getSectionName(da);
  }

  /// \brief Gets or creates a section.
  virtual AtomSection<HexagonELFType> *
  createSection(StringRef name, int32_t contentType,
                DefinedAtom::ContentPermissions contentPermissions,
                Layout::SectionOrder sectionOrder) {
    if ((contentType == DefinedAtom::typeDataFast) ||
       (contentType == DefinedAtom::typeZeroFillFast))
      return _sdataSection;
    return DefaultLayout<HexagonELFType>::createSection(
        name, contentType, contentPermissions, sectionOrder);
  }

  /// \brief get the segment type for the section thats defined by the target
  virtual Layout::SegmentType
  getSegmentType(Section<HexagonELFType> *section) const {
    if (section->order() == ORDER_SDATA)
      return PT_LOAD;

    return DefaultLayout<HexagonELFType>::getSegmentType(section);
  }

  Section<HexagonELFType> *getSDataSection() const {
    return _sdataSection;
  }

private:
  llvm::BumpPtrAllocator _alloc;
  SDataSection<HexagonELFType> *_sdataSection;
};

/// \brief TargetHandler for Hexagon
class HexagonTargetHandler LLVM_FINAL :
    public DefaultTargetHandler<HexagonELFType> {
public:
  HexagonTargetHandler(HexagonLinkingContext &targetInfo);

  virtual void registerRelocationNames(Registry &registry);

  bool doesOverrideELFHeader() { return true; }

  void setELFHeader(ELFHeader<HexagonELFType> *elfHeader) {
    elfHeader->e_ident(llvm::ELF::EI_VERSION, 1);
    elfHeader->e_ident(llvm::ELF::EI_OSABI, 0);
    elfHeader->e_version(1);
    elfHeader->e_flags(0x3);
  }

  virtual HexagonTargetLayout<HexagonELFType> &targetLayout() {
    return _targetLayout;
  }

  virtual const HexagonTargetRelocationHandler &getRelocationHandler() const {
    return _relocationHandler;
  }

  void addDefaultAtoms() {
    _hexagonRuntimeFile->addAbsoluteAtom("_SDA_BASE_");
    if (_context.isDynamic()) {
      _hexagonRuntimeFile->addAbsoluteAtom("_GLOBAL_OFFSET_TABLE_");
      _hexagonRuntimeFile->addAbsoluteAtom("_DYNAMIC");
    }
  }

  virtual bool
  createImplicitFiles(std::vector<std::unique_ptr<File> > &result) {
    // Add the default atoms as defined for hexagon
    addDefaultAtoms();
    result.push_back(std::move(_hexagonRuntimeFile));
    return true;
  }

  void finalizeSymbolValues() {
    auto sdabaseAtomIter = _targetLayout.findAbsoluteAtom("_SDA_BASE_");
    (*sdabaseAtomIter)->_virtualAddr =
        _targetLayout.getSDataSection()->virtualAddr();
    if (_context.isDynamic()) {
      auto gotAtomIter =
          _targetLayout.findAbsoluteAtom("_GLOBAL_OFFSET_TABLE_");
      _gotSymAtom = (*gotAtomIter);
      auto gotpltSection = _targetLayout.findOutputSection(".got.plt");
      if (gotpltSection)
        _gotSymAtom->_virtualAddr = gotpltSection->virtualAddr();
      else
        _gotSymAtom->_virtualAddr = 0;
      auto dynamicAtomIter = _targetLayout.findAbsoluteAtom("_DYNAMIC");
      auto dynamicSection = _targetLayout.findOutputSection(".dynamic");
      if (dynamicSection)
        (*dynamicAtomIter)->_virtualAddr = dynamicSection->virtualAddr();
      else
        (*dynamicAtomIter)->_virtualAddr = 0;
    }
  }

  uint64_t getGOTSymAddr() const {
    if (!_gotSymAtom) return 0;
    return _gotSymAtom->_virtualAddr;
  }

  virtual std::unique_ptr<Reader> getObjReader(bool atomizeStrings) {
    return std::unique_ptr<Reader>(new HexagonELFObjectReader(atomizeStrings));
  }

  virtual std::unique_ptr<Reader> getDSOReader(bool useShlibUndefines) {
    return std::unique_ptr<Reader>(new HexagonELFDSOReader(useShlibUndefines));
  }

private:
  static const Registry::KindStrings kindStrings[];

  HexagonTargetLayout<HexagonELFType> _targetLayout;
  HexagonTargetRelocationHandler _relocationHandler;
  std::unique_ptr<HexagonRuntimeFile<HexagonELFType> > _hexagonRuntimeFile;
  AtomLayout *_gotSymAtom;
};
} // end namespace elf
} // end namespace lld

#endif
