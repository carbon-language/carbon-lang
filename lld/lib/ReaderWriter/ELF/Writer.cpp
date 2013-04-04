//===- lib/ReaderWriter/ELF/WriterELF.cpp ---------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/Writer.h"

#include "DynamicLibraryWriter.h"
#include "ExecutableWriter.h"


using namespace llvm;
using namespace llvm::object;
namespace lld {

std::unique_ptr<Writer> createWriterELF(const ELFTargetInfo &info) {
  using llvm::object::ELFType;
  // Set the default layout to be the static executable layout
  // We would set the layout to a dynamic executable layout
  // if we came across any shared libraries in the process

  switch(info.getOutputType()) {
  case llvm::ELF::ET_EXEC:
    if (info.is64Bits()) {
      if (info.isLittleEndian()) 
        return std::unique_ptr<Writer>(new
            elf::ExecutableWriter<ELFType<support::little, 8, true>>(info));
      else
        return std::unique_ptr<Writer>(new
                elf::ExecutableWriter<ELFType<support::big, 8, true>>(info));
    } else {
      if (info.isLittleEndian()) 
        return std::unique_ptr<Writer>(new
            elf::ExecutableWriter<ELFType<support::little, 4, false>>(info));
      else
        return std::unique_ptr<Writer>(new
                elf::ExecutableWriter<ELFType<support::big, 4, false>>(info));
    }
  break;
  case llvm::ELF::ET_DYN:
    if (info.is64Bits()) {
      if (info.isLittleEndian()) 
        return std::unique_ptr<Writer>(new
          elf::DynamicLibraryWriter<ELFType<support::little, 8, true>>(info));
      else
        return std::unique_ptr<Writer>(new
              elf::DynamicLibraryWriter<ELFType<support::big, 8, true>>(info));
    } else {
      if (info.isLittleEndian()) 
        return std::unique_ptr<Writer>(new
          elf::DynamicLibraryWriter<ELFType<support::little, 4, false>>(info));
      else
        return std::unique_ptr<Writer>(new
              elf::DynamicLibraryWriter<ELFType<support::big, 4, false>>(info));
    }
  break;
  case llvm::ELF::ET_REL:
    llvm_unreachable("TODO: support -r mode");
  default:
    llvm_unreachable("unsupported output type");
  }
}

} // namespace lld
