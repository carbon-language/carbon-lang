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

class AArch64ExecutableWriter : public ExecutableWriter<ELF64LE> {
public:
  AArch64ExecutableWriter(AArch64LinkingContext &ctx,
                          TargetLayout<ELF64LE> &layout);

protected:
  // Add any runtime files and their atoms to the output
  void createImplicitFiles(std::vector<std::unique_ptr<File>> &) override;
};

AArch64ExecutableWriter::AArch64ExecutableWriter(AArch64LinkingContext &ctx,
                                                 TargetLayout<ELF64LE> &layout)
    : ExecutableWriter<ELF64LE>(ctx, layout) {}

void AArch64ExecutableWriter::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  ExecutableWriter<ELF64LE>::createImplicitFiles(result);
  auto gotFile = llvm::make_unique<SimpleFile>("GOTFile");
  gotFile->addAtom(*new (gotFile->allocator()) GlobalOffsetTableAtom(*gotFile));
  if (this->_ctx.isDynamic())
    gotFile->addAtom(*new (gotFile->allocator()) DynamicAtom(*gotFile));
  result.push_back(std::move(gotFile));
}

} // namespace elf
} // namespace lld

#endif
