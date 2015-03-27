//===- lib/ReaderWriter/ELF/AArch64/AArch64DynamicLibraryWriter.h ---------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef AARCH64_DYNAMIC_LIBRARY_WRITER_H
#define AARCH64_DYNAMIC_LIBRARY_WRITER_H

#include "AArch64LinkingContext.h"
#include "AArch64TargetHandler.h"
#include "DynamicLibraryWriter.h"

namespace lld {
namespace elf {

template <class ELFT>
class AArch64DynamicLibraryWriter : public DynamicLibraryWriter<ELFT> {
public:
  AArch64DynamicLibraryWriter(AArch64LinkingContext &ctx,
                              AArch64TargetLayout<ELFT> &layout);

protected:
  // Add any runtime files and their atoms to the output
  virtual bool createImplicitFiles(std::vector<std::unique_ptr<File>> &);

  virtual void finalizeDefaultAtomValues() {
    return DynamicLibraryWriter<ELFT>::finalizeDefaultAtomValues();
  }

  virtual void addDefaultAtoms() {
    return DynamicLibraryWriter<ELFT>::addDefaultAtoms();
  }

private:
  class GOTFile : public SimpleFile {
  public:
    GOTFile(const ELFLinkingContext &eti) : SimpleFile("GOTFile") {}
    llvm::BumpPtrAllocator _alloc;
  };

  std::unique_ptr<GOTFile> _gotFile;
  AArch64LinkingContext &_ctx;
  AArch64TargetLayout<ELFT> &_AArch64Layout;
};

template <class ELFT>
AArch64DynamicLibraryWriter<ELFT>::AArch64DynamicLibraryWriter(
    AArch64LinkingContext &ctx, AArch64TargetLayout<ELFT> &layout)
    : DynamicLibraryWriter<ELFT>(ctx, layout), _gotFile(new GOTFile(ctx)),
      _ctx(ctx), _AArch64Layout(layout) {}

template <class ELFT>
bool AArch64DynamicLibraryWriter<ELFT>::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  DynamicLibraryWriter<ELFT>::createImplicitFiles(result);
  _gotFile->addAtom(*new (_gotFile->_alloc) GlobalOffsetTableAtom(*_gotFile));
  _gotFile->addAtom(*new (_gotFile->_alloc) DynamicAtom(*_gotFile));
  result.push_back(std::move(_gotFile));
  return true;
}

} // namespace elf
} // namespace lld

#endif
