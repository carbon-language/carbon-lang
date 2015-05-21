//===--------- lib/ReaderWriter/ELF/ARM/ARMExecutableWriter.h -------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_ARM_ARM_EXECUTABLE_WRITER_H
#define LLD_READER_WRITER_ELF_ARM_ARM_EXECUTABLE_WRITER_H

#include "ExecutableWriter.h"
#include "ARMELFWriters.h"
#include "ARMLinkingContext.h"
#include "ARMTargetHandler.h"

namespace lld {
namespace elf {

class ARMExecutableWriter : public ARMELFWriter<ExecutableWriter<ELF32LE>> {
public:
  ARMExecutableWriter(ARMLinkingContext &ctx, ARMTargetLayout &layout);

protected:
  // Add any runtime files and their atoms to the output
  void createImplicitFiles(std::vector<std::unique_ptr<File>> &) override;

  void processUndefinedSymbol(StringRef symName,
                              RuntimeFile<ELF32LE> &file) const override;

private:
  ARMLinkingContext &_ctx;
};

ARMExecutableWriter::ARMExecutableWriter(ARMLinkingContext &ctx,
                                         ARMTargetLayout &layout)
    : ARMELFWriter(ctx, layout), _ctx(ctx) {}

void ARMExecutableWriter::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  ExecutableWriter::createImplicitFiles(result);
  // Add default atoms for ARM.
  if (_ctx.isDynamic()) {
    auto file = llvm::make_unique<RuntimeFile<ELF32LE>>(_ctx, "ARM exec file");
    file->addAbsoluteAtom(gotSymbol);
    file->addAbsoluteAtom(dynamicSymbol);
    result.push_back(std::move(file));
  }
}

void ARMExecutableWriter::processUndefinedSymbol(
    StringRef symName, RuntimeFile<ELF32LE> &file) const {
  if (symName == gotSymbol) {
    file.addAbsoluteAtom(gotSymbol);
  } else if (symName.startswith("__exidx")) {
    file.addAbsoluteAtom("__exidx_start");
    file.addAbsoluteAtom("__exidx_end");
  } else if (symName == "__ehdr_start") {
    file.addAbsoluteAtom("__ehdr_start");
  }
}

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_ELF_ARM_ARM_EXECUTABLE_WRITER_H
