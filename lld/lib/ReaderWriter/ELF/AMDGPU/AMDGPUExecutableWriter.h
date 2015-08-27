//===- lib/ReaderWriter/ELF/AMDGPU/AMDGPUExecutableWriter.h ---------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef AMDGPU_EXECUTABLE_WRITER_H
#define AMDGPU_EXECUTABLE_WRITER_H

#include "ExecutableWriter.h"
#include "AMDGPULinkingContext.h"
#include "AMDGPUSymbolTable.h"
#include "AMDGPUTargetHandler.h"

namespace lld {
namespace elf {

class AMDGPUTargetLayout;

class AMDGPUExecutableWriter : public ExecutableWriter<ELF64LE> {
public:
  AMDGPUExecutableWriter(AMDGPULinkingContext &ctx, AMDGPUTargetLayout &layout);

  unique_bump_ptr<SymbolTable<ELF64LE>> createSymbolTable() override {
    return unique_bump_ptr<SymbolTable<ELF64LE>>(new (this->_alloc)
                                                     AMDGPUSymbolTable(_ctx));
  }

  void createImplicitFiles(std::vector<std::unique_ptr<File>> &Result) override;
  void finalizeDefaultAtomValues() override;

private:
  AMDGPULinkingContext &_ctx;
};

} // namespace elf
} // namespace lld

#endif // AMDGPU_EXECUTABLE_WRITER_H
