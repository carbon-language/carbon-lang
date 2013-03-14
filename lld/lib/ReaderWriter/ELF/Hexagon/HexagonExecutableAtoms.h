//===- lib/ReaderWriter/ELF/Hexagon/HexagonExecutableAtoms.h --------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_HEXAGON_EXECUTABLE_ATOM_H
#define LLD_READER_WRITER_ELF_HEXAGON_EXECUTABLE_ATOM_H

#include "ExecutableAtoms.h"

namespace lld {
namespace elf {
typedef llvm::object::ELFType<llvm::support::little, 4, false> HexagonELFType;
class HexagonTargetInfo;

template <class HexagonELFType> class HexagonRuntimeFile
  : public CRuntimeFile<HexagonELFType> {
public:
  HexagonRuntimeFile(const HexagonTargetInfo &hti)
    :CRuntimeFile<HexagonELFType>(hti, "Hexagon runtime file")
  {}

};
} // elf
} // lld

#endif // LLD_READER_WRITER_ELF_HEXAGON_EXECUTABLE_ATOM_H
