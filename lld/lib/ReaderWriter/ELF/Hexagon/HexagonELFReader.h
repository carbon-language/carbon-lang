//===- lib/ReaderWriter/ELF/HexagonELFReader.h ----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_HEXAGON_ELF_READER_H
#define LLD_READER_WRITER_HEXAGON_ELF_READER_H

#include "ELFReader.h"
#include "HexagonELFFile.h"

namespace lld {
namespace elf {

typedef llvm::object::ELFType<llvm::support::little, 2, false> HexagonELFType;

typedef ELFObjectReader<HexagonELFType, HexagonLinkingContext,
                        lld::elf::HexagonELFFile> HexagonELFObjectReader;
typedef ELFDSOReader<HexagonELFType, HexagonLinkingContext> HexagonELFDSOReader;

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_ELF_READER_H
