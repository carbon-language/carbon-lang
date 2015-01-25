//===- lld/ReaderWriter/ELF/Mips/MipsRelocationHandler.h ------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_MIPS_MIPS_RELOCATION_HANDLER_H
#define LLD_READER_WRITER_ELF_MIPS_MIPS_RELOCATION_HANDLER_H

namespace lld {
namespace elf {

class MipsLinkingContext;
class TargetRelocationHandler;

template <class ELFT>
std::unique_ptr<TargetRelocationHandler>
createMipsRelocationHandler(MipsLinkingContext &ctx);

} // elf
} // lld

#endif
