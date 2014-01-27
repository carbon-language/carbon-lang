//===- lib/ReaderWriter/ELF/Mips/MipsTargetHandler.h ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_MIPS_TARGET_HANDLER_H
#define LLD_READER_WRITER_ELF_MIPS_TARGET_HANDLER_H

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

class MipsDynamicTable : public DynamicTable<Mips32ElELFType> {
public:
  MipsDynamicTable(MipsLinkingContext &context,
                   MipsTargetLayout<Mips32ElELFType> &layout)
      : DynamicTable<Mips32ElELFType>(
            context, layout, ".dynamic",
            DefaultLayout<Mips32ElELFType>::ORDER_DYNAMIC),
        _mipsTargetLayout(layout) {}

  virtual void createDefaultEntries() {
    DynamicTable<Mips32ElELFType>::createDefaultEntries();

    Elf_Dyn dyn;

    // Version id for the Runtime Linker Interface.
    dyn.d_un.d_val = 1;
    dyn.d_tag = DT_MIPS_RLD_VERSION;
    addEntry(dyn);

    // MIPS flags.
    dyn.d_un.d_val = RHF_NOTPOT;
    dyn.d_tag = DT_MIPS_FLAGS;
    addEntry(dyn);

    // The base address of the segment.
    dyn.d_un.d_ptr = 0;
    dyn.d_tag = DT_MIPS_BASE_ADDRESS;
    _dt_baseaddr = addEntry(dyn);

    // Number of local global offset table entries.
    dyn.d_un.d_val = 0;
    dyn.d_tag = DT_MIPS_LOCAL_GOTNO;
    _dt_localgot = addEntry(dyn);

    // Number of entries in the .dynsym section.
    dyn.d_un.d_val = 0;
    dyn.d_tag = DT_MIPS_SYMTABNO;
    _dt_symtabno = addEntry(dyn);

    // The index of the first dynamic symbol table entry that corresponds
    // to an entry in the global offset table.
    dyn.d_un.d_val = 0;
    dyn.d_tag = DT_MIPS_GOTSYM;
    _dt_gotsym = addEntry(dyn);

    // Address of the .got section.
    dyn.d_un.d_val = 0;
    dyn.d_tag = DT_PLTGOT;
    _dt_pltgot = addEntry(dyn);
  }

  virtual void updateDynamicTable() {
    DynamicTable<Mips32ElELFType>::updateDynamicTable();

    // Assign the minimum segment address to the DT_MIPS_BASE_ADDRESS tag.
    auto baseAddr = std::numeric_limits<uint64_t>::max();
    for (auto si : _mipsTargetLayout.segments())
      if (si->segmentType() != llvm::ELF::PT_NULL)
        baseAddr = std::min(baseAddr, si->virtualAddr());
    _entries[_dt_baseaddr].d_un.d_val = baseAddr;

    auto &got = _mipsTargetLayout.getGOTSection();

    _entries[_dt_symtabno].d_un.d_val = getSymbolTable()->size();
    _entries[_dt_gotsym].d_un.d_val =
        getSymbolTable()->size() - got.getGlobalCount();
    _entries[_dt_localgot].d_un.d_val = got.getLocalCount();
    _entries[_dt_pltgot].d_un.d_ptr =
        _mipsTargetLayout.findOutputSection(".got")->virtualAddr();
  }

private:
  std::size_t _dt_symtabno;
  std::size_t _dt_localgot;
  std::size_t _dt_gotsym;
  std::size_t _dt_pltgot;
  std::size_t _dt_baseaddr;
  MipsTargetLayout<Mips32ElELFType> &_mipsTargetLayout;
};
} // end namespace elf
} // end namespace lld

#endif
