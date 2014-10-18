//===- lib/ReaderWriter/ELF/AArch64/AArch64ExecutableWriter.h -------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef AARCH64_EXECUTABLE_WRITER_H
#define AARCH64_EXECUTABLE_WRITER_H

#include "AArch64LinkingContext.h"
#include "ExecutableWriter.h"

namespace lld {
namespace elf {

template <class ELFT>
class AArch64ExecutableWriter : public ExecutableWriter<ELFT> {
public:
  AArch64ExecutableWriter(AArch64LinkingContext &context,
                          AArch64TargetLayout<ELFT> &layout);

protected:
  // Add any runtime files and their atoms to the output
  bool createImplicitFiles(std::vector<std::unique_ptr<File>> &) override;

  void finalizeDefaultAtomValues() override {
    return ExecutableWriter<ELFT>::finalizeDefaultAtomValues();
  }

  void addDefaultAtoms() override{
    return ExecutableWriter<ELFT>::addDefaultAtoms();
  }

private:
  class GOTFile : public SimpleFile {
  public:
    GOTFile(const ELFLinkingContext &eti) : SimpleFile("GOTFile") {}
    llvm::BumpPtrAllocator _alloc;
  };

  std::unique_ptr<GOTFile> _gotFile;
  AArch64LinkingContext &_context;
  AArch64TargetLayout<ELFT> &_AArch64Layout;
};

template <class ELFT>
AArch64ExecutableWriter<ELFT>::AArch64ExecutableWriter(
    AArch64LinkingContext &context, AArch64TargetLayout<ELFT> &layout)
    : ExecutableWriter<ELFT>(context, layout), _gotFile(new GOTFile(context)),
      _context(context), _AArch64Layout(layout) {}

template <class ELFT>
bool AArch64ExecutableWriter<ELFT>::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  ExecutableWriter<ELFT>::createImplicitFiles(result);
  _gotFile->addAtom(*new (_gotFile->_alloc) GLOBAL_OFFSET_TABLEAtom(*_gotFile));
  if (_context.isDynamic())
    _gotFile->addAtom(*new (_gotFile->_alloc) DYNAMICAtom(*_gotFile));
  result.push_back(std::move(_gotFile));
  return true;
}

} // namespace elf
} // namespace lld

#endif
