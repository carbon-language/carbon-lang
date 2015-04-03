//===- lib/ReaderWriter/ELF/AArch64/AArch64ELFReader.h --------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_AARCH64_AARCH64_ELF_READER_H
#define LLD_READER_WRITER_AARCH64_AARCH64_ELF_READER_H

#include "ELFReader.h"

namespace lld {
namespace elf {

class AArch64LinkingContext;

typedef llvm::object::ELFType<llvm::support::little, 2, true> AArch64ELFType;

typedef ELFObjectReader<AArch64ELFType, AArch64LinkingContext,
                        lld::elf::ELFFile> AArch64ELFObjectReader;

typedef ELFDSOReader<AArch64ELFType, AArch64LinkingContext> AArch64ELFDSOReader;

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_AARCH64_AARCH64_ELF_READER_H
