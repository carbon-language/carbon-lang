//===- lib/ReaderWriter/ELF/Hexagon/HexagonExecutableAtoms.h --------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_HEXAGON_HEXAGON_EXECUTABLE_ATOM_H
#define LLD_READER_WRITER_ELF_HEXAGON_HEXAGON_EXECUTABLE_ATOM_H

#include "ELFFile.h"

namespace lld {
class ELFLinkingContext;

namespace elf {
typedef llvm::object::ELFType<llvm::support::little, 2, false> ELFT;

class HexagonRuntimeFile : public RuntimeFile<ELFT> {
public:
  HexagonRuntimeFile(ELFLinkingContext &ctx)
      : RuntimeFile<ELFT>(ctx, "Hexagon runtime file") {}
};
} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_HEXAGON_HEXAGON_EXECUTABLE_ATOM_H
