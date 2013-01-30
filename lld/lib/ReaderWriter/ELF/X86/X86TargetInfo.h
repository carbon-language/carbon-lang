//===- lib/ReaderWriter/ELF/X86/X86TargetInfo.h ---------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_X86_TARGETINFO_H
#define LLD_READER_WRITER_ELF_X86_TARGETINFO_H

#include "X86TargetHandler.h"

#include "lld/Core/LinkerOptions.h"
#include "lld/ReaderWriter/ELFTargetInfo.h"

#include "llvm/Object/ELF.h"
#include "llvm/Support/ELF.h"

namespace lld {
namespace elf {
class X86TargetInfo LLVM_FINAL : public ELFTargetInfo {
public:
  X86TargetInfo(const LinkerOptions &lo) : ELFTargetInfo(lo) {
    _targetHandler = std::unique_ptr<TargetHandlerBase>(
        new X86TargetHandler(*this));
  }

  virtual uint64_t getPageSize() const { return 0x1000; }
};
} // end namespace elf
} // end namespace lld
#endif
