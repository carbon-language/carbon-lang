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
  AArch64ExecutableWriter(AArch64LinkingContext &ctx,
                          TargetLayout<ELFT> &layout);

protected:
  // Add any runtime files and their atoms to the output
  void createImplicitFiles(std::vector<std::unique_ptr<File>> &) override;

  void finalizeDefaultAtomValues() override {
    return ExecutableWriter<ELFT>::finalizeDefaultAtomValues();
  }

private:
  class GOTFile : public SimpleFile {
  public:
    GOTFile(const ELFLinkingContext &eti) : SimpleFile("GOTFile") {}
    llvm::BumpPtrAllocator _alloc;
  };

  std::unique_ptr<GOTFile> _gotFile;
  AArch64LinkingContext &_ctx;
  TargetLayout<ELFT> &_layout;
};

template <class ELFT>
AArch64ExecutableWriter<ELFT>::AArch64ExecutableWriter(
    AArch64LinkingContext &ctx, TargetLayout<ELFT> &layout)
    : ExecutableWriter<ELFT>(ctx, layout), _gotFile(new GOTFile(ctx)),
      _ctx(ctx), _layout(layout) {}

template <class ELFT>
void AArch64ExecutableWriter<ELFT>::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  ExecutableWriter<ELFT>::createImplicitFiles(result);
  _gotFile->addAtom(*new (_gotFile->_alloc) GlobalOffsetTableAtom(*_gotFile));
  if (_ctx.isDynamic())
    _gotFile->addAtom(*new (_gotFile->_alloc) DynamicAtom(*_gotFile));
  result.push_back(std::move(_gotFile));
}

} // namespace elf
} // namespace lld

#endif
