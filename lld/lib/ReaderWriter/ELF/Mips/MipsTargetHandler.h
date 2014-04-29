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

namespace lld {
namespace elf {

/// \brief TargetLayout for Mips
template <class ELFType>
class MipsTargetLayout final : public TargetLayout<ELFType> {
public:
  MipsTargetLayout(const MipsLinkingContext &ctx)
      : TargetLayout<ELFType>(ctx),
        _gotSection(new (_alloc) MipsGOTSection<ELFType>(ctx)),
        _pltSection(new (_alloc) MipsPLTSection<ELFType>(ctx)),
        _cachedGP(false) {}

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

  /// \brief GP offset relative to .got section.
  uint64_t getGPOffset() const { return 0x7FF0; }

  /// \brief Get the cached value of the GP atom.
  AtomLayout *getGP() {
    if (!_cachedGP) {
      auto gpAtomIter = this->findAbsoluteAtom("_gp_disp");
      _gp = *(gpAtomIter);
      _cachedGP = true;
    }
    return _gp;
  }

private:
  llvm::BumpPtrAllocator _alloc;
  MipsGOTSection<ELFType> *_gotSection;
  MipsPLTSection<ELFType> *_pltSection;
  AtomLayout *_gp;
  bool _cachedGP;
};

/// \brief Mips Runtime file.
template <class ELFType>
class MipsRuntimeFile final : public CRuntimeFile<ELFType> {
public:
  MipsRuntimeFile(const MipsLinkingContext &context)
      : CRuntimeFile<ELFType>(context, "Mips runtime file") {}
};

/// \brief TargetHandler for Mips
class MipsTargetHandler final : public DefaultTargetHandler<Mips32ElELFType> {
public:
  MipsTargetHandler(MipsLinkingContext &context);

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
  MipsLinkingContext &_context;
  std::unique_ptr<MipsRuntimeFile<Mips32ElELFType>> _runtimeFile;
  std::unique_ptr<MipsTargetLayout<Mips32ElELFType>> _targetLayout;
  std::unique_ptr<MipsTargetRelocationHandler> _relocationHandler;
};

template <class ELFT>
class MipsDynamicSymbolTable : public DynamicSymbolTable<ELFT> {
public:
  MipsDynamicSymbolTable(const MipsLinkingContext &context,
                         MipsTargetLayout<ELFT> &layout)
      : DynamicSymbolTable<ELFT>(
            context, layout, ".dynsym",
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
