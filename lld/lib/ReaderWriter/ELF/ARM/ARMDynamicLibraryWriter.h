//===- lib/ReaderWriter/ELF/ARM/ARMDynamicLibraryWriter.h -----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_ARM_ARM_DYNAMIC_LIBRARY_WRITER_H
#define LLD_READER_WRITER_ELF_ARM_ARM_DYNAMIC_LIBRARY_WRITER_H

#include "DynamicLibraryWriter.h"
#include "ARMELFWriters.h"
#include "ARMLinkingContext.h"
#include "ARMTargetHandler.h"

namespace lld {
namespace elf {

class ARMDynamicLibraryWriter
    : public ARMELFWriter<DynamicLibraryWriter<ELF32LE>> {
public:
  ARMDynamicLibraryWriter(ARMLinkingContext &ctx, ARMTargetLayout &layout);

protected:
  // Add any runtime files and their atoms to the output
  void createImplicitFiles(std::vector<std::unique_ptr<File>> &) override;

private:
  ARMLinkingContext &_ctx;
};

ARMDynamicLibraryWriter::ARMDynamicLibraryWriter(ARMLinkingContext &ctx,
                                                 ARMTargetLayout &layout)
    : ARMELFWriter(ctx, layout), _ctx(ctx) {}

void ARMDynamicLibraryWriter::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  DynamicLibraryWriter::createImplicitFiles(result);
  auto file = llvm::make_unique<RuntimeFile<ELF32LE>>(_ctx, "ARM dynamic file");
  file->addAbsoluteAtom(gotSymbol);
  file->addAbsoluteAtom(dynamicSymbol);
  result.push_back(std::move(file));
}

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_ELF_ARM_ARM_DYNAMIC_LIBRARY_WRITER_H
