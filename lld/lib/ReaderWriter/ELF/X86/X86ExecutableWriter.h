//===- lib/ReaderWriter/ELF/X86/X86ExecutableWriter.h ---------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef X86_X86_EXECUTABLE_WRITER_H
#define X86_X86_EXECUTABLE_WRITER_H

#include "ExecutableWriter.h"
#include "X86LinkingContext.h"

namespace lld {
namespace elf {

class X86ExecutableWriter : public ExecutableWriter<ELF32LE> {
public:
  X86ExecutableWriter(X86LinkingContext &ctx, TargetLayout<ELF32LE> &layout);

protected:
  // Add any runtime files and their atoms to the output
  void createImplicitFiles(std::vector<std::unique_ptr<File>> &) override;
};

X86ExecutableWriter::X86ExecutableWriter(X86LinkingContext &ctx,
                                         TargetLayout<ELF32LE> &layout)
    : ExecutableWriter(ctx, layout) {}

void X86ExecutableWriter::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  ExecutableWriter::createImplicitFiles(result);
}

} // namespace elf
} // namespace lld

#endif
