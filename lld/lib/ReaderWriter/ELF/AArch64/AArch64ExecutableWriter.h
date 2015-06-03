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

#include "ExecutableWriter.h"

namespace lld {
namespace elf {

class AArch64TargetLayout;
class AArch64LinkingContext;

class AArch64ExecutableWriter : public ExecutableWriter<ELF64LE> {
public:
  AArch64ExecutableWriter(AArch64LinkingContext &ctx,
                          AArch64TargetLayout &layout);

protected:
  // Add any runtime files and their atoms to the output
  void createImplicitFiles(std::vector<std::unique_ptr<File>> &) override;

  void buildDynamicSymbolTable(const File &file) override;

private:
  AArch64TargetLayout &_targetLayout;
};

} // namespace elf
} // namespace lld

#endif
