//===- lib/ReaderWriter/ELF/Mips/MipsDynamicLibraryWriter.h ---------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_MIPS_DYNAMIC_LIBRARY_WRITER_H
#define LLD_READER_WRITER_ELF_MIPS_DYNAMIC_LIBRARY_WRITER_H

#include "DynamicLibraryWriter.h"
#include "MipsDynamicTable.h"
#include "MipsELFWriters.h"
#include "MipsLinkingContext.h"

namespace lld {
namespace elf {

template <typename ELFT> class MipsTargetLayout;

template <class ELFT>
class MipsDynamicLibraryWriter : public DynamicLibraryWriter<ELFT> {
public:
  MipsDynamicLibraryWriter(MipsLinkingContext &ctx,
                           MipsTargetLayout<ELFT> &layout);

protected:
  // Add any runtime files and their atoms to the output
  bool createImplicitFiles(std::vector<std::unique_ptr<File>> &) override;

  void finalizeDefaultAtomValues() override;

  std::error_code setELFHeader() override {
    DynamicLibraryWriter<ELFT>::setELFHeader();
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
MipsDynamicLibraryWriter<ELFT>::MipsDynamicLibraryWriter(
    MipsLinkingContext &ctx, MipsTargetLayout<ELFT> &layout)
    : DynamicLibraryWriter<ELFT>(ctx, layout), _writeHelper(ctx, layout),
      _mipsContext(ctx), _mipsTargetLayout(layout) {}

template <class ELFT>
bool MipsDynamicLibraryWriter<ELFT>::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  DynamicLibraryWriter<ELFT>::createImplicitFiles(result);
  result.push_back(std::move(_writeHelper.createRuntimeFile()));
  return true;
}

template <class ELFT>
void MipsDynamicLibraryWriter<ELFT>::finalizeDefaultAtomValues() {
  // Finalize the atom values that are part of the parent.
  DynamicLibraryWriter<ELFT>::finalizeDefaultAtomValues();
  _writeHelper.finalizeMipsRuntimeAtomValues();
}

/// \brief create dynamic table
template <class ELFT>
LLD_UNIQUE_BUMP_PTR(DynamicTable<ELFT>)
    MipsDynamicLibraryWriter<ELFT>::createDynamicTable() {
  return LLD_UNIQUE_BUMP_PTR(DynamicTable<ELFT>)(new (
      this->_alloc) MipsDynamicTable<ELFT>(_mipsContext, _mipsTargetLayout));
}

/// \brief create dynamic symbol table
template <class ELFT>
LLD_UNIQUE_BUMP_PTR(DynamicSymbolTable<ELFT>)
    MipsDynamicLibraryWriter<ELFT>::createDynamicSymbolTable() {
  return LLD_UNIQUE_BUMP_PTR(
      DynamicSymbolTable<ELFT>)(new (this->_alloc) MipsDynamicSymbolTable<ELFT>(
      _mipsContext, _mipsTargetLayout));
}

} // namespace elf
} // namespace lld

#endif
