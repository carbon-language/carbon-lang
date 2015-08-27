//===- lld/ReaderWriter/ELF/AMDGPU/AMDGPURelocationHandler.h --------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_AMDGPU_AMDGPU_RELOCATION_HANDLER_H
#define LLD_READER_WRITER_ELF_AMDGPU_AMDGPU_RELOCATION_HANDLER_H

#include "lld/ReaderWriter/ELFLinkingContext.h"
#include <system_error>

namespace lld {
namespace elf {
class AMDGPUTargetHandler;
class AMDGPUTargetLayout;

class AMDGPUTargetRelocationHandler final : public TargetRelocationHandler {
public:
  AMDGPUTargetRelocationHandler(AMDGPUTargetLayout &layout) { }

  std::error_code applyRelocation(ELFWriter &, llvm::FileOutputBuffer &,
                                  const AtomLayout &,
                                  const Reference &) const override;

};
} // elf
} // lld
#endif
