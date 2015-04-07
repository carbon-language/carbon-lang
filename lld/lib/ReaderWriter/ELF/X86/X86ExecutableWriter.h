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

template <class ELFT>
class X86ExecutableWriter : public ExecutableWriter<ELFT> {
public:
  X86ExecutableWriter(X86LinkingContext &ctx, TargetLayout<ELFT> &layout);

protected:
  // Add any runtime files and their atoms to the output
  void createImplicitFiles(std::vector<std::unique_ptr<File>> &) override;
};

template <class ELFT>
X86ExecutableWriter<ELFT>::X86ExecutableWriter(X86LinkingContext &ctx,
                                               TargetLayout<ELFT> &layout)
    : ExecutableWriter<ELFT>(ctx, layout) {}

template <class ELFT>
void X86ExecutableWriter<ELFT>::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  ExecutableWriter<ELFT>::createImplicitFiles(result);
}

} // namespace elf
} // namespace lld

#endif
