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
class MipsExecutableWriter : public ExecutableWriter<ELFT>,
                             public MipsELFWriter<ELFT> {
public:
  MipsExecutableWriter(MipsLinkingContext &context,
                       MipsTargetLayout<ELFT> &layout);

protected:
  // Add any runtime files and their atoms to the output
  virtual bool createImplicitFiles(std::vector<std::unique_ptr<File>> &);

  virtual void finalizeDefaultAtomValues();

  virtual error_code setELFHeader() {
    ExecutableWriter<ELFT>::setELFHeader();
    MipsELFWriter<ELFT>::setELFHeader(*this->_elfHeader);
    return error_code::success();
  }

  LLD_UNIQUE_BUMP_PTR(DynamicTable<ELFT>) createDynamicTable();

  LLD_UNIQUE_BUMP_PTR(DynamicSymbolTable<ELFT>) createDynamicSymbolTable();

private:
  void addDefaultAtoms() {
    if (this->_context.isDynamic()) {
      _mipsRuntimeFile->addAbsoluteAtom("_GLOBAL_OFFSET_TABLE_");
      _mipsRuntimeFile->addAbsoluteAtom("_gp");
      _mipsRuntimeFile->addAbsoluteAtom("_gp_disp");
    }
  }

  std::unique_ptr<MipsRuntimeFile<ELFT>> _mipsRuntimeFile;
  MipsLinkingContext &_mipsContext;
  MipsTargetLayout<Mips32ElELFType> &_mipsTargetLayout;
};

template <class ELFT>
MipsExecutableWriter<ELFT>::MipsExecutableWriter(MipsLinkingContext &context,
                                                 MipsTargetLayout<ELFT> &layout)
    : ExecutableWriter<ELFT>(context, layout),
      MipsELFWriter<ELFT>(context, layout),
      _mipsRuntimeFile(new MipsRuntimeFile<ELFT>(context)),
      _mipsContext(context), _mipsTargetLayout(layout) {}

template <class ELFT>
bool MipsExecutableWriter<ELFT>::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  ExecutableWriter<ELFT>::createImplicitFiles(result);
  // Add the default atoms as defined for mips
  addDefaultAtoms();
  result.push_back(std::move(_mipsRuntimeFile));
  return true;
}

template <class ELFT>
void MipsExecutableWriter<ELFT>::finalizeDefaultAtomValues() {
  // Finalize the atom values that are part of the parent.
  ExecutableWriter<ELFT>::finalizeDefaultAtomValues();
  MipsELFWriter<ELFT>::finalizeMipsRuntimeAtomValues();
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
  return LLD_UNIQUE_BUMP_PTR(DynamicSymbolTable<ELFT>)(new (
      this->_alloc) MipsDynamicSymbolTable(_mipsContext, _mipsTargetLayout));
}

} // namespace elf
} // namespace lld

#endif
