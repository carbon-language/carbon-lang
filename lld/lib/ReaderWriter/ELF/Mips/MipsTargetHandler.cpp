//===- lib/ReaderWriter/ELF/Mips/MipsTargetHandler.cpp --------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "File.h"
#include "MipsLinkingContext.h"
#include "MipsTargetHandler.h"

using namespace lld;
using namespace elf;

namespace {

class MipsDynamicSymbolTable : public DynamicSymbolTable<Mips32ElELFType> {
public:
  MipsDynamicSymbolTable(const MipsLinkingContext &context)
      : DynamicSymbolTable<Mips32ElELFType>(
            context, ".dynsym",
            DefaultLayout<Mips32ElELFType>::ORDER_DYNAMIC_SYMBOLS),
        _layout(context.getTargetLayout()) {}

  virtual void sortSymbols() {
    std::stable_sort(_symbolTable.begin(), _symbolTable.end(),
                     [this](const SymbolEntry &A, const SymbolEntry &B) {
      if (A._symbol.getBinding() != STB_GLOBAL &&
          B._symbol.getBinding() != STB_GLOBAL)
        return A._symbol.getBinding() < B._symbol.getBinding();

      return _layout.getGOTSection().compare(A._atom, B._atom);
    });
  }

private:
  const MipsTargetLayout<Mips32ElELFType> &_layout;
};

class MipsDynamicTable : public DynamicTable<Mips32ElELFType> {
public:
  MipsDynamicTable(MipsLinkingContext &context)
      : DynamicTable<Mips32ElELFType>(
            context, ".dynamic", DefaultLayout<Mips32ElELFType>::ORDER_DYNAMIC),
        _layout(context.getTargetLayout()) {}

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
    addEntry(dyn);

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

    auto &got = _layout.getGOTSection();

    _entries[_dt_symtabno].d_un.d_val = getSymbolTable()->size();
    _entries[_dt_gotsym].d_un.d_val =
        getSymbolTable()->size() - got.getGlobalCount();
    _entries[_dt_localgot].d_un.d_val = got.getLocalCount();
    _entries[_dt_pltgot].d_un.d_ptr =
        _layout.findOutputSection(".got")->virtualAddr();
  }

private:
  MipsTargetLayout<Mips32ElELFType> &_layout;

  std::size_t _dt_symtabno;
  std::size_t _dt_localgot;
  std::size_t _dt_gotsym;
  std::size_t _dt_pltgot;
};
}

MipsTargetHandler::MipsTargetHandler(MipsLinkingContext &context)
    : DefaultTargetHandler(context), _targetLayout(context),
      _relocationHandler(context, *this) {}

uint64_t MipsTargetHandler::getGPDispSymAddr() const {
  return _gpDispSymAtom ? _gpDispSymAtom->_virtualAddr : 0;
}

MipsTargetLayout<Mips32ElELFType> &MipsTargetHandler::targetLayout() {
  return _targetLayout;
}

const MipsTargetRelocationHandler &
MipsTargetHandler::getRelocationHandler() const {
  return _relocationHandler;
}

LLD_UNIQUE_BUMP_PTR(DynamicTable<Mips32ElELFType>)
MipsTargetHandler::createDynamicTable() {
  return LLD_UNIQUE_BUMP_PTR(DynamicTable<Mips32ElELFType>)(
      new (_alloc) MipsDynamicTable(
          static_cast<MipsLinkingContext &>(_context)));
}

LLD_UNIQUE_BUMP_PTR(DynamicSymbolTable<Mips32ElELFType>)
MipsTargetHandler::createDynamicSymbolTable() {
  return LLD_UNIQUE_BUMP_PTR(DynamicSymbolTable<Mips32ElELFType>)(
      new (_alloc) MipsDynamicSymbolTable(
          static_cast<MipsLinkingContext &>(_context)));
}

bool MipsTargetHandler::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  typedef CRuntimeFile<Mips32ElELFType> RFile;
  auto file = std::unique_ptr<RFile>(new RFile(_context, "MIPS runtime file"));

  if (_context.isDynamic()) {
    file->addAbsoluteAtom("_GLOBAL_OFFSET_TABLE_");
    file->addAbsoluteAtom("_gp_disp");
  }
  result.push_back(std::move(file));
  return true;
}

void MipsTargetHandler::finalizeSymbolValues() {
  DefaultTargetHandler<Mips32ElELFType>::finalizeSymbolValues();

  if (_context.isDynamic()) {
    auto gotSection = _targetLayout.findOutputSection(".got");

    auto gotAtomIter = _targetLayout.findAbsoluteAtom("_GLOBAL_OFFSET_TABLE_");
    assert(gotAtomIter != _targetLayout.absoluteAtoms().end());
    _gotSymAtom = (*gotAtomIter);
    _gotSymAtom->_virtualAddr = gotSection ? gotSection->virtualAddr() : 0;

    auto gpDispAtomIter = _targetLayout.findAbsoluteAtom("_gp_disp");
    assert(gpDispAtomIter != _targetLayout.absoluteAtoms().end());
    _gpDispSymAtom = (*gpDispAtomIter);
    _gpDispSymAtom->_virtualAddr =
        gotSection ? gotSection->virtualAddr() + 0x7FF0 : 0;
  }
}
