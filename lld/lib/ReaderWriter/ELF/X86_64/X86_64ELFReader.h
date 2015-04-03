//===- lib/ReaderWriter/ELF/X86_64/X86_64ELFReader.h ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_X86_64_X86_64_ELF_READER_H
#define LLD_READER_WRITER_X86_64_X86_64_ELF_READER_H

#include "ELFReader.h"

namespace lld {
namespace elf {

class X86_64LinkingContext;
typedef llvm::object::ELFType<llvm::support::little, 2, true> X86_64ELFType;

typedef ELFObjectReader<X86_64ELFType, X86_64LinkingContext, lld::elf::ELFFile>
    X86_64ELFObjectReader;
typedef ELFDSOReader<X86_64ELFType, X86_64LinkingContext> X86_64ELFDSOReader;

} // namespace elf
} // namespace lld

#endif // LLD_READER_WRITER_ELF_X86_64_X86_64_READER_H
