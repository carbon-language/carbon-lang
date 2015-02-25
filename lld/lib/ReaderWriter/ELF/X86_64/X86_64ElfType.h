//===- lib/ReaderWriter/ELF/X86_64/X86_64ElfType.h ------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_X86_64_X86_64_ELF_TYPE_H
#define LLD_READER_WRITER_ELF_X86_64_X86_64_ELF_TYPE_H

#include "llvm/Object/ELF.h"

namespace lld {
namespace elf {
typedef llvm::object::ELFType<llvm::support::little, 2, true> X86_64ELFType;
}
}

#endif
