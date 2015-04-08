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
};

template <class ELFT>
X86DynamicLibraryWriter<ELFT>::X86DynamicLibraryWriter(
    X86LinkingContext &ctx, TargetLayout<ELFT> &layout)
    : DynamicLibraryWriter<ELFT>(ctx, layout) {}

template <class ELFT>
void X86DynamicLibraryWriter<ELFT>::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  DynamicLibraryWriter<ELFT>::createImplicitFiles(result);
  auto gotFile = llvm::make_unique<SimpleFile>("GOTFile");
  gotFile->addAtom(*new (gotFile->allocator()) GlobalOffsetTableAtom(*gotFile));
  gotFile->addAtom(*new (gotFile->allocator()) DynamicAtom(*gotFile));
  result.push_back(std::move(gotFile));
}

} // namespace elf
} // namespace lld

#endif
