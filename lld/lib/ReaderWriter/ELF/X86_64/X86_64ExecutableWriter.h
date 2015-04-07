//===- lib/ReaderWriter/ELF/X86/X86_64ExecutableWriter.h ------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef X86_64_EXECUTABLE_WRITER_H
#define X86_64_EXECUTABLE_WRITER_H

#include "ExecutableWriter.h"
#include "X86_64ElfType.h"
#include "X86_64LinkingContext.h"

namespace lld {
namespace elf {

class X86_64ExecutableWriter : public ExecutableWriter<X86_64ELFType> {
public:
  X86_64ExecutableWriter(X86_64LinkingContext &ctx, X86_64TargetLayout &layout)
      : ExecutableWriter(ctx, layout), _gotFile(new GOTFile(ctx)), _ctx(ctx) {}

protected:
  // Add any runtime files and their atoms to the output
  void
  createImplicitFiles(std::vector<std::unique_ptr<File>> &result) override {
    ExecutableWriter::createImplicitFiles(result);
    _gotFile->addAtom(*new (_gotFile->_alloc)
                      GlobalOffsetTableAtom(*_gotFile));
    if (_ctx.isDynamic())
      _gotFile->addAtom(*new (_gotFile->_alloc) DynamicAtom(*_gotFile));
    result.push_back(std::move(_gotFile));
  }

  void finalizeDefaultAtomValues() override {
    return ExecutableWriter::finalizeDefaultAtomValues();
  }

private:
  class GOTFile : public SimpleFile {
  public:
    GOTFile(const ELFLinkingContext &eti) : SimpleFile("GOTFile") {}
    llvm::BumpPtrAllocator _alloc;
  };

  std::unique_ptr<GOTFile> _gotFile;
  X86_64LinkingContext &_ctx;
};

} // namespace elf
} // namespace lld

#endif
