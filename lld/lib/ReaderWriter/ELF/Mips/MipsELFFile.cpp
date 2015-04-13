//===- lib/ReaderWriter/ELF/Mips/MipsELFFile.cpp --------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MipsELFFile.h"

namespace lld {
namespace elf {

MIPSFILE_INSTANTIATION(Mips32ELType)
MIPSFILE_INSTANTIATION(Mips64ELType)
MIPSFILE_INSTANTIATION(Mips32BEType)
MIPSFILE_INSTANTIATION(Mips64BEType)

}
}
