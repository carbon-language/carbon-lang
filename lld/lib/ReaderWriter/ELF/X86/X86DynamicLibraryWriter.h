//===- lib/ReaderWriter/ELF/X86/X86DynamicLibraryWriter.h -----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef X86_X86_DYNAMIC_LIBRARY_WRITER_H
#define X86_X86_DYNAMIC_LIBRARY_WRITER_H

#include "DynamicLibraryWriter.h"
#include "X86LinkingContext.h"

namespace lld {
namespace elf {

template <class ELFT>
class X86DynamicLibraryWriter : public DynamicLibraryWriter<ELFT> {
public:
  X86DynamicLibraryWriter(X86LinkingContext &ctx, TargetLayout<ELFT> &layout);

protected:
  // Add any runtime files and their atoms to the output
  void createImplicitFiles(std::vector<std::unique_ptr<File>> &) override;

  void finalizeDefaultAtomValues() override {
    return DynamicLibraryWriter<ELFT>::finalizeDefaultAtomValues();
  }

  void addDefaultAtoms() override {
    return DynamicLibraryWriter<ELFT>::addDefaultAtoms();
  }

private:
  class GOTFile : public SimpleFile {
  public:
    GOTFile(const ELFLinkingContext &eti) : SimpleFile("GOTFile") {}
    llvm::BumpPtrAllocator _alloc;
  };

  std::unique_ptr<GOTFile> _gotFile;
  X86LinkingContext &_ctx;
  TargetLayout<ELFT> &_layout;
};

template <class ELFT>
X86DynamicLibraryWriter<ELFT>::X86DynamicLibraryWriter(
    X86LinkingContext &ctx, TargetLayout<ELFT> &layout)
    : DynamicLibraryWriter<ELFT>(ctx, layout), _gotFile(new GOTFile(ctx)),
      _ctx(ctx), _layout(layout) {}

template <class ELFT>
void X86DynamicLibraryWriter<ELFT>::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  DynamicLibraryWriter<ELFT>::createImplicitFiles(result);
  _gotFile->addAtom(*new (_gotFile->_alloc) GlobalOffsetTableAtom(*_gotFile));
  _gotFile->addAtom(*new (_gotFile->_alloc) DynamicAtom(*_gotFile));
  result.push_back(std::move(_gotFile));
}

} // namespace elf
} // namespace lld

#endif
