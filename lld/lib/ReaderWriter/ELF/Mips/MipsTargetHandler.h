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
#include "MipsLinkingContext.h"
#include "MipsRelocationHandler.h"
#include "MipsSectionChunks.h"
#include "TargetLayout.h"

namespace lld {
namespace elf {

/// \brief TargetLayout for Mips
template <class ELFType>
class MipsTargetLayout LLVM_FINAL : public TargetLayout<ELFType> {
public:
  MipsTargetLayout(const MipsLinkingContext &ctx)
      : TargetLayout<ELFType>(ctx),
        _gotSection(new (_alloc) MipsGOTSection<ELFType>(ctx)),
        _cachedGP(false) {}

  const MipsGOTSection<ELFType> &getGOTSection() const { return *_gotSection; }

  virtual AtomSection<ELFType> *
  createSection(StringRef name, int32_t type,
                DefinedAtom::ContentPermissions permissions,
                Layout::SectionOrder order) {
    if (type == DefinedAtom::typeGOT && name == ".got")
      return _gotSection;
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
  AtomLayout *_gp;
  bool _cachedGP;
};

/// \brief Mips Runtime file.
template <class ELFType> class MipsRuntimeFile : public CRuntimeFile<ELFType> {
public:
  MipsRuntimeFile(const MipsLinkingContext &context)
      : CRuntimeFile<ELFType>(context, "Mips runtime file") {}
};

/// \brief TargetHandler for Mips
class MipsTargetHandler LLVM_FINAL
    : public DefaultTargetHandler<Mips32ElELFType> {
public:
  MipsTargetHandler(MipsLinkingContext &context);

  virtual MipsTargetLayout<Mips32ElELFType> &getTargetLayout() {
    return *(_mipsTargetLayout.get());
  }

  virtual const MipsTargetRelocationHandler &getRelocationHandler() const {
    return *(_mipsRelocationHandler.get());
  }

  virtual std::unique_ptr<Writer> getWriter();

  virtual void registerRelocationNames(Registry &registry);

private:
  static const Registry::KindStrings kindStrings[];
  MipsLinkingContext &_mipsLinkingContext;
  std::unique_ptr<MipsRuntimeFile<Mips32ElELFType>> _mipsRuntimeFile;
  std::unique_ptr<MipsTargetLayout<Mips32ElELFType>> _mipsTargetLayout;
  std::unique_ptr<MipsTargetRelocationHandler> _mipsRelocationHandler;
};

class MipsDynamicSymbolTable : public DynamicSymbolTable<Mips32ElELFType> {
public:
  MipsDynamicSymbolTable(const MipsLinkingContext &context,
                         MipsTargetLayout<Mips32ElELFType> &layout)
      : DynamicSymbolTable<Mips32ElELFType>(
            context, layout, ".dynsym",
            DefaultLayout<Mips32ElELFType>::ORDER_DYNAMIC_SYMBOLS),
        _mipsTargetLayout(layout) {}

  virtual void sortSymbols() {
    std::stable_sort(_symbolTable.begin(), _symbolTable.end(),
                     [this](const SymbolEntry &A, const SymbolEntry &B) {
      if (A._symbol.getBinding() != STB_GLOBAL &&
          B._symbol.getBinding() != STB_GLOBAL)
        return A._symbol.getBinding() < B._symbol.getBinding();

      return _mipsTargetLayout.getGOTSection().compare(A._atom, B._atom);
    });
  }

private:
  MipsTargetLayout<Mips32ElELFType> &_mipsTargetLayout;
};

} // end namespace elf
} // end namespace lld

#endif
