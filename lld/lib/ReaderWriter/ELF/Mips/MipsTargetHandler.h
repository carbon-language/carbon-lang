//===- lib/ReaderWriter/ELF/Mips/MipsTargetHandler.h ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_MIPS_MIPS_TARGET_HANDLER_H
#define LLD_READER_WRITER_ELF_MIPS_MIPS_TARGET_HANDLER_H

#include "DefaultTargetHandler.h"
#include "MipsELFReader.h"
#include "MipsLinkingContext.h"
#include "MipsRelocationHandler.h"
#include "MipsSectionChunks.h"
#include "TargetLayout.h"

#include "llvm/ADT/DenseSet.h"

namespace lld {
namespace elf {

/// \brief TargetLayout for Mips
template <class ELFType>
class MipsTargetLayout final : public TargetLayout<ELFType> {
public:
  MipsTargetLayout(const MipsLinkingContext &ctx)
      : TargetLayout<ELFType>(ctx),
        _gotSection(new (_alloc) MipsGOTSection<ELFType>(ctx)),
        _pltSection(new (_alloc) MipsPLTSection<ELFType>(ctx)) {}

  const MipsGOTSection<ELFType> &getGOTSection() const { return *_gotSection; }
  const MipsPLTSection<ELFType> &getPLTSection() const { return *_pltSection; }

  AtomSection<ELFType> *
  createSection(StringRef name, int32_t type,
                DefinedAtom::ContentPermissions permissions,
                Layout::SectionOrder order) override {
    if (type == DefinedAtom::typeGOT && name == ".got")
      return _gotSection;
    if (type == DefinedAtom::typeStub && name == ".plt")
      return _pltSection;
    return DefaultLayout<ELFType>::createSection(name, type, permissions,
                                                 order);
  }

  StringRef getSectionName(const DefinedAtom *da) const override {
    return llvm::StringSwitch<StringRef>(da->customSectionName())
        .StartsWith(".ctors", ".ctors")
        .StartsWith(".dtors", ".dtors")
        .Default(TargetLayout<ELFType>::getSectionName(da));
  }

  Layout::SegmentType getSegmentType(Section<ELFType> *section) const override {
    switch (section->order()) {
    case DefaultLayout<ELFType>::ORDER_CTORS:
    case DefaultLayout<ELFType>::ORDER_DTORS:
      return llvm::ELF::PT_LOAD;
    default:
      return TargetLayout<ELFType>::getSegmentType(section);
    }
  }

  ErrorOr<const lld::AtomLayout &> addAtom(const Atom *atom) override {
    // Maintain:
    // 1. Set of shared library atoms referenced by regular defined atoms.
    // 2. Set of shared library atoms have corresponding R_MIPS_COPY copies.
    if (const auto *da = dyn_cast<DefinedAtom>(atom))
      for (const Reference *ref : *da) {
        if (ref->kindNamespace() == lld::Reference::KindNamespace::ELF) {
          assert(ref->kindArch() == Reference::KindArch::Mips);
          if (ref->kindValue() == llvm::ELF::R_MIPS_COPY)
            _copiedDynSymNames.insert(atom->name());
        }
      }

    return TargetLayout<ELFType>::addAtom(atom);
  }

  bool isCopied(const SharedLibraryAtom *sla) const {
    return _copiedDynSymNames.count(sla->name());
  }

  /// \brief GP offset relative to .got section.
  uint64_t getGPOffset() const { return 0x7FF0; }

  /// \brief Get '_gp' symbol atom layout.
  AtomLayout *getGP() {
    if (!_gpAtom.hasValue()) {
      auto atom = this->findAbsoluteAtom("_gp");
      _gpAtom = atom != this->absoluteAtoms().end() ? *atom : nullptr;
    }
    return *_gpAtom;
  }

  /// \brief Get '_gp_disp' symbol atom layout.
  AtomLayout *getGPDisp() {
    if (!_gpDispAtom.hasValue()) {
      auto atom = this->findAbsoluteAtom("_gp_disp");
      _gpDispAtom = atom != this->absoluteAtoms().end() ? *atom : nullptr;
    }
    return *_gpDispAtom;
  }

private:
  llvm::BumpPtrAllocator _alloc;
  MipsGOTSection<ELFType> *_gotSection;
  MipsPLTSection<ELFType> *_pltSection;
  llvm::Optional<AtomLayout *> _gpAtom;
  llvm::Optional<AtomLayout *> _gpDispAtom;
  llvm::StringSet<> _copiedDynSymNames;
};

/// \brief Mips Runtime file.
template <class ELFType>
class MipsRuntimeFile final : public CRuntimeFile<ELFType> {
public:
  MipsRuntimeFile(const MipsLinkingContext &ctx)
      : CRuntimeFile<ELFType>(ctx, "Mips runtime file") {}
};

/// \brief TargetHandler for Mips
class MipsTargetHandler final : public DefaultTargetHandler<Mips32ElELFType> {
public:
  MipsTargetHandler(MipsLinkingContext &ctx);

  MipsTargetLayout<Mips32ElELFType> &getTargetLayout() override {
    return *_targetLayout;
  }

  std::unique_ptr<Reader> getObjReader(bool atomizeStrings) override {
    return std::unique_ptr<Reader>(new MipsELFObjectReader(atomizeStrings));
  }

  const MipsTargetRelocationHandler &getRelocationHandler() const override {
    return *_relocationHandler;
  }

  std::unique_ptr<Writer> getWriter() override;

  void registerRelocationNames(Registry &registry) override;

private:
  static const Registry::KindStrings kindStrings[];
  MipsLinkingContext &_ctx;
  std::unique_ptr<MipsRuntimeFile<Mips32ElELFType>> _runtimeFile;
  std::unique_ptr<MipsTargetLayout<Mips32ElELFType>> _targetLayout;
  std::unique_ptr<MipsTargetRelocationHandler> _relocationHandler;
};

template <class ELFT>
class MipsDynamicSymbolTable : public DynamicSymbolTable<ELFT> {
public:
  MipsDynamicSymbolTable(const MipsLinkingContext &ctx,
                         MipsTargetLayout<ELFT> &layout)
      : DynamicSymbolTable<ELFT>(ctx, layout, ".dynsym",
                                 DefaultLayout<ELFT>::ORDER_DYNAMIC_SYMBOLS),
        _targetLayout(layout) {}

  void sortSymbols() override {
    typedef typename DynamicSymbolTable<ELFT>::SymbolEntry SymbolEntry;
    std::stable_sort(this->_symbolTable.begin(), this->_symbolTable.end(),
                     [this](const SymbolEntry &A, const SymbolEntry &B) {
      if (A._symbol.getBinding() != STB_GLOBAL &&
          B._symbol.getBinding() != STB_GLOBAL)
        return A._symbol.getBinding() < B._symbol.getBinding();

      return _targetLayout.getGOTSection().compare(A._atom, B._atom);
    });
  }

  void finalize() override {
    const auto &pltSection = _targetLayout.getPLTSection();

    // Under some conditions a dynamic symbol table record should hold a symbol
    // value of the corresponding PLT entry. For details look at the PLT entry
    // creation code in the class MipsRelocationPass. Let's update atomLayout
    // fields for such symbols.
    for (auto &ste : this->_symbolTable) {
      if (!ste._atom || ste._atomLayout)
        continue;
      auto *layout = pltSection.findPLTLayout(ste._atom);
      if (layout) {
        ste._symbol.st_value = layout->_virtualAddr;
        ste._symbol.st_other |= ELF::STO_MIPS_PLT;
      }
    }

    DynamicSymbolTable<Mips32ElELFType>::finalize();
  }

private:
  MipsTargetLayout<ELFT> &_targetLayout;
};

} // end namespace elf
} // end namespace lld

#endif
