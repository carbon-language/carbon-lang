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

#include "lld/Core/Reference.h"

namespace lld {
namespace elf {

class MipsLinkingContext;
template<typename ELFT> class MipsTargetLayout;

template <class ELFT>
std::unique_ptr<TargetRelocationHandler>
createMipsRelocationHandler(MipsLinkingContext &ctx,
                            MipsTargetLayout<ELFT> &layout);

template <class ELFT>
Reference::Addend readMipsRelocAddend(Reference::KindValue kind,
                                      const uint8_t *content);
} // elf
} // lld

#endif
