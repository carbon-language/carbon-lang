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
#include "X86_64LinkingContext.h"

namespace lld {
namespace elf {

class X86_64ExecutableWriter : public ExecutableWriter<ELF64LE> {
public:
  X86_64ExecutableWriter(X86_64LinkingContext &ctx, X86_64TargetLayout &layout)
      : ExecutableWriter(ctx, layout) {}

protected:
  // Add any runtime files and their atoms to the output
  void
  createImplicitFiles(std::vector<std::unique_ptr<File>> &result) override {
    ExecutableWriter::createImplicitFiles(result);
    auto gotFile = llvm::make_unique<SimpleFile>("GOTFile");
    gotFile->addAtom(*new (gotFile->allocator())
                         GlobalOffsetTableAtom(*gotFile));
    if (this->_ctx.isDynamic())
      gotFile->addAtom(*new (gotFile->allocator()) DynamicAtom(*gotFile));
    result.push_back(std::move(gotFile));
  }
};

} // namespace elf
} // namespace lld

#endif
