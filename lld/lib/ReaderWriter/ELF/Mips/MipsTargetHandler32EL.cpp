//===- lib/ReaderWriter/ELF/Mips/MipsTargetHandler32EL.cpp ----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MipsTargetHandler.h"

using namespace lld;
using namespace lld::elf;

std::unique_ptr<TargetHandler>
lld::elf::createMips32ELTargetHandler(MipsLinkingContext &ctx) {
  return llvm::make_unique<MipsTargetHandler<Mips32ELType>>(ctx);
}
