//===- lib/ReaderWriter/ELF/Mips/MipsExecutableWriter.h -------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_MIPS_MIPS_EXECUTABLE_WRITER_H
#define LLD_READER_WRITER_ELF_MIPS_MIPS_EXECUTABLE_WRITER_H

#include "ExecutableWriter.h"
#include "MipsDynamicTable.h"
#include "MipsELFWriters.h"
#include "MipsLinkingContext.h"

namespace lld {
namespace elf {

template <typename ELFT> class MipsTargetLayout;

template <class ELFT>
class MipsExecutableWriter : public ExecutableWriter<ELFT> {
public:
  MipsExecutableWriter(MipsLinkingContext &ctx, MipsTargetLayout<ELFT> &layout);

protected:
  void buildDynamicSymbolTable(const File &file) override;

  // Add any runtime files and their atoms to the output
  bool createImplicitFiles(std::vector<std::unique_ptr<File>> &) override;

  void finalizeDefaultAtomValues() override;

  std::error_code setELFHeader() override {
    ExecutableWriter<ELFT>::setELFHeader();
    _writeHelper.setELFHeader(*this->_elfHeader);
    return std::error_code();
  }

  bool isNeededTagRequired(const SharedLibraryAtom *sla) const override {
    return _writeHelper.isNeededTagRequired(sla);
  }

  LLD_UNIQUE_BUMP_PTR(DynamicTable<ELFT>) createDynamicTable();

  LLD_UNIQUE_BUMP_PTR(DynamicSymbolTable<ELFT>) createDynamicSymbolTable();

private:
  MipsELFWriter<ELFT> _writeHelper;
  MipsLinkingContext &_mipsContext;
  MipsTargetLayout<Mips32ElELFType> &_mipsTargetLayout;
};

template <class ELFT>
MipsExecutableWriter<ELFT>::MipsExecutableWriter(MipsLinkingContext &ctx,
                                                 MipsTargetLayout<ELFT> &layout)
    : ExecutableWriter<ELFT>(ctx, layout), _writeHelper(ctx, layout),
      _mipsContext(ctx), _mipsTargetLayout(layout) {}

template <class ELFT>
void MipsExecutableWriter<ELFT>::buildDynamicSymbolTable(const File &file) {
  // MIPS ABI requires to add to dynsym even undefined symbols
  // if they have a corresponding entries in a global part of GOT.
  for (auto sec : this->_layout.sections())
    if (auto section = dyn_cast<AtomSection<ELFT>>(sec))
      for (const auto &atom : section->atoms()) {
        if (_writeHelper.hasGlobalGOTEntry(atom->_atom))
          this->_dynamicSymbolTable->addSymbol(atom->_atom, section->ordinal(),
                                               atom->_virtualAddr, atom);
      }

  for (const UndefinedAtom *a : file.undefined())
    // FIXME (simon): Consider to move this check to the
    // MipsELFUndefinedAtom class method. That allows to
    // handle more complex coditions in the future.
    if (_writeHelper.hasGlobalGOTEntry(a))
      this->_dynamicSymbolTable->addSymbol(a, ELF::SHN_UNDEF);

  ExecutableWriter<ELFT>::buildDynamicSymbolTable(file);
}

template <class ELFT>
bool MipsExecutableWriter<ELFT>::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  ExecutableWriter<ELFT>::createImplicitFiles(result);
  result.push_back(std::move(_writeHelper.createRuntimeFile()));
  return true;
}

template <class ELFT>
void MipsExecutableWriter<ELFT>::finalizeDefaultAtomValues() {
  // Finalize the atom values that are part of the parent.
  ExecutableWriter<ELFT>::finalizeDefaultAtomValues();
  _writeHelper.finalizeMipsRuntimeAtomValues();
}

/// \brief create dynamic table
template <class ELFT>
LLD_UNIQUE_BUMP_PTR(DynamicTable<ELFT>)
    MipsExecutableWriter<ELFT>::createDynamicTable() {
  return LLD_UNIQUE_BUMP_PTR(DynamicTable<ELFT>)(new (
      this->_alloc) MipsDynamicTable<ELFT>(_mipsContext, _mipsTargetLayout));
}

/// \brief create dynamic symbol table
template <class ELFT>
LLD_UNIQUE_BUMP_PTR(DynamicSymbolTable<ELFT>)
    MipsExecutableWriter<ELFT>::createDynamicSymbolTable() {
  return LLD_UNIQUE_BUMP_PTR(
      DynamicSymbolTable<ELFT>)(new (this->_alloc) MipsDynamicSymbolTable<ELFT>(
      _mipsContext, _mipsTargetLayout));
}

} // namespace elf
} // namespace lld

#endif
