//===- lib/ReaderWriter/ELF/AMDGPU/AMDGPURelocationHandler.cpp -----------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AMDGPURelocationHandler.h"

using namespace lld;
using namespace lld::elf;

std::error_code AMDGPUTargetRelocationHandler::applyRelocation(
    ELFWriter &writer, llvm::FileOutputBuffer &buf, const AtomLayout &atom,
    const Reference &ref) const {
  return std::error_code();
}
