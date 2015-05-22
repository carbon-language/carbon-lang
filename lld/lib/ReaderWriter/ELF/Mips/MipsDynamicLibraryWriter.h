//===- lib/ReaderWriter/ELF/Mips/MipsDynamicLibraryWriter.h ---------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_MIPS_MIPS_DYNAMIC_LIBRARY_WRITER_H
#define LLD_READER_WRITER_ELF_MIPS_MIPS_DYNAMIC_LIBRARY_WRITER_H

#include "DynamicLibraryWriter.h"
#include "MipsDynamicTable.h"
#include "MipsELFWriters.h"
#include "MipsLinkingContext.h"
#include "MipsSectionChunks.h"

namespace lld {
namespace elf {

template <typename ELFT> class MipsSymbolTable;
template <typename ELFT> class MipsDynamicSymbolTable;
template <typename ELFT> class MipsTargetLayout;

template <class ELFT>
class MipsDynamicLibraryWriter : public DynamicLibraryWriter<ELFT> {
public:
  MipsDynamicLibraryWriter(MipsLinkingContext &ctx,
                           MipsTargetLayout<ELFT> &layout);

protected:
  // Add any runtime files and their atoms to the output
  void createImplicitFiles(std::vector<std::unique_ptr<File>> &) override;

  void finalizeDefaultAtomValues() override;
  void createDefaultSections() override;

  std::error_code setELFHeader() override {
    DynamicLibraryWriter<ELFT>::setELFHeader();
    _writeHelper.setELFHeader(*this->_elfHeader);
    return std::error_code();
  }

  unique_bump_ptr<SymbolTable<ELFT>> createSymbolTable() override;
  unique_bump_ptr<DynamicTable<ELFT>> createDynamicTable() override;
  unique_bump_ptr<DynamicSymbolTable<ELFT>> createDynamicSymbolTable() override;

private:
  MipsELFWriter<ELFT> _writeHelper;
  MipsTargetLayout<ELFT> &_targetLayout;
  unique_bump_ptr<Section<ELFT>> _reginfo;
};

template <class ELFT>
MipsDynamicLibraryWriter<ELFT>::MipsDynamicLibraryWriter(
    MipsLinkingContext &ctx, MipsTargetLayout<ELFT> &layout)
    : DynamicLibraryWriter<ELFT>(ctx, layout), _writeHelper(ctx, layout),
      _targetLayout(layout) {}

template <class ELFT>
void MipsDynamicLibraryWriter<ELFT>::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  DynamicLibraryWriter<ELFT>::createImplicitFiles(result);
  result.push_back(_writeHelper.createRuntimeFile());
}

template <class ELFT>
void MipsDynamicLibraryWriter<ELFT>::finalizeDefaultAtomValues() {
  // Finalize the atom values that are part of the parent.
  DynamicLibraryWriter<ELFT>::finalizeDefaultAtomValues();
  _writeHelper.finalizeMipsRuntimeAtomValues();
}

template <class ELFT>
void MipsDynamicLibraryWriter<ELFT>::createDefaultSections() {
  DynamicLibraryWriter<ELFT>::createDefaultSections();
  const auto &ctx = static_cast<const MipsLinkingContext &>(this->_ctx);
  const auto &mask = ctx.getMergedReginfoMask();
  if (!mask.hasValue())
    return;
  if (ELFT::Is64Bits)
    _reginfo = unique_bump_ptr<Section<ELFT>>(
        new (this->_alloc) MipsOptionsSection<ELFT>(ctx, _targetLayout, *mask));
  else
    _reginfo = unique_bump_ptr<Section<ELFT>>(
        new (this->_alloc) MipsReginfoSection<ELFT>(ctx, _targetLayout, *mask));
  this->_layout.addSection(_reginfo.get());
}

template <class ELFT>
unique_bump_ptr<SymbolTable<ELFT>>
    MipsDynamicLibraryWriter<ELFT>::createSymbolTable() {
  return unique_bump_ptr<SymbolTable<ELFT>>(
      new (this->_alloc) MipsSymbolTable<ELFT>(this->_ctx));
}

/// \brief create dynamic table
template <class ELFT>
unique_bump_ptr<DynamicTable<ELFT>>
    MipsDynamicLibraryWriter<ELFT>::createDynamicTable() {
  return unique_bump_ptr<DynamicTable<ELFT>>(
      new (this->_alloc) MipsDynamicTable<ELFT>(this->_ctx, _targetLayout));
}

/// \brief create dynamic symbol table
template <class ELFT>
unique_bump_ptr<DynamicSymbolTable<ELFT>>
    MipsDynamicLibraryWriter<ELFT>::createDynamicSymbolTable() {
  return unique_bump_ptr<DynamicSymbolTable<ELFT>>(
      new (this->_alloc)
          MipsDynamicSymbolTable<ELFT>(this->_ctx, _targetLayout));
}

} // namespace elf
} // namespace lld

#endif
