//===- lib/ReaderWriter/ELF/AMDGPU/AMDGPULinkingContext.h ---------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_AMDGPU_AMDGPU_LINKING_CONTEXT_H
#define LLD_READER_WRITER_ELF_AMDGPU_AMDGPU_LINKING_CONTEXT_H

#include "OutputELFWriter.h"
#include "lld/ReaderWriter/ELFLinkingContext.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/ELF.h"

namespace lld {
namespace elf {

class AMDGPULinkingContext final : public ELFLinkingContext {
public:
  AMDGPULinkingContext(llvm::Triple triple);
  int getMachineType() const override { return llvm::ELF::EM_AMDGPU; }

  void registerRelocationNames(Registry &r) override;

  StringRef entrySymbolName() const override;
};

void setAMDGPUELFHeader(ELFHeader<ELF64LE> &elfHeader);

} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_AMDGPU_AMDGPU_LINKING_CONTEXT_H
