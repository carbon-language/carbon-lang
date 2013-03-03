//===- lib/ReaderWriter/ELF/Writer.cpp ------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "ExecutableWriter.h"

using namespace llvm;
using namespace llvm::object;

namespace lld {

std::unique_ptr<Writer> createWriterELF(const ELFTargetInfo &TI) {
  using llvm::object::ELFType;
  // Set the default layout to be the static executable layout
  // We would set the layout to a dynamic executable layout
  // if we came across any shared libraries in the process

  if (!TI.is64Bits() && TI.isLittleEndian())
    return std::unique_ptr<Writer>(new
        elf::ExecutableWriter<ELFType<support::little, 4, false>>(TI));
  else if (TI.is64Bits() && TI.isLittleEndian())
    return std::unique_ptr<Writer>(new
        elf::ExecutableWriter<ELFType<support::little, 8, true>>(TI));
  else if (!TI.is64Bits() && !TI.isLittleEndian())
    return std::unique_ptr<Writer>(new
        elf::ExecutableWriter<ELFType<support::big, 4, false>>(TI));
  else if (TI.is64Bits() && !TI.isLittleEndian())
    return std::unique_ptr<Writer>(new
        elf::ExecutableWriter<ELFType<support::big, 8, true>>(TI));

  llvm_unreachable("Invalid Options!");
}

}
