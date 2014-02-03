//===- lib/ReaderWriter/ELF/Mips/MipsRelocationPass.h ---------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_MIPS_MIPS_RELOCATION_PASS_H
#define LLD_READER_WRITER_ELF_MIPS_MIPS_RELOCATION_PASS_H

#include <memory>

namespace lld {
class Pass;

namespace elf {
class MipsLinkingContext;

std::unique_ptr<Pass> createMipsRelocationPass(MipsLinkingContext &ctx);

} // end namespace elf
} // end namespace lld

#endif
