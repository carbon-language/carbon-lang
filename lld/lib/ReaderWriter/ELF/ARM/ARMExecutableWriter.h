//===--------- lib/ReaderWriter/ELF/ARM/ARMExecutableWriter.h -------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_ARM_ARM_EXECUTABLE_WRITER_H
#define LLD_READER_WRITER_ELF_ARM_ARM_EXECUTABLE_WRITER_H

#include "ExecutableWriter.h"
#include "ARMLinkingContext.h"
#include "ARMTargetHandler.h"
#include "ARMSymbolTable.h"

namespace lld {
namespace elf {

template <class ELFT>
class ARMExecutableWriter : public ExecutableWriter<ELFT> {
public:
  ARMExecutableWriter(ARMLinkingContext &context,
                      ARMTargetLayout<ELFT> &layout);

protected:
  // Add any runtime files and their atoms to the output
  bool createImplicitFiles(std::vector<std::unique_ptr<File>> &) override;

  void finalizeDefaultAtomValues() override {
    // Finalize the atom values that are part of the parent.
    ExecutableWriter<ELFT>::finalizeDefaultAtomValues();
  }

  void addDefaultAtoms() override {
    ExecutableWriter<ELFT>::addDefaultAtoms();
  }

  /// \brief Create symbol table.
  unique_bump_ptr<SymbolTable<ELFT>> createSymbolTable() override;

private:
  ARMLinkingContext &_context;
  ARMTargetLayout<ELFT> &_armLayout;
};

template <class ELFT>
ARMExecutableWriter<ELFT>::ARMExecutableWriter(ARMLinkingContext &context,
                                               ARMTargetLayout<ELFT> &layout)
    : ExecutableWriter<ELFT>(context, layout), _context(context),
      _armLayout(layout) {}

template <class ELFT>
bool ARMExecutableWriter<ELFT>::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  ExecutableWriter<ELFT>::createImplicitFiles(result);
  return true;
}

template <class ELFT>
unique_bump_ptr<SymbolTable<ELFT>>
    ARMExecutableWriter<ELFT>::createSymbolTable() {
  return unique_bump_ptr<SymbolTable<ELFT>>(
      new (this->_alloc) ARMSymbolTable<ELFT>(this->_context));
}

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_ELF_ARM_ARM_EXECUTABLE_WRITER_H
