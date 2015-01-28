//===- lld/ReaderWriter/ELFTargets.h --------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_TARGETS_H
#define LLD_READER_WRITER_ELF_TARGETS_H

#include "ELFLinkingContext.h"

namespace lld {
namespace elf {

#define LLVM_TARGET(TargetName) \
  class TargetName##LinkingContext final : public ELFLinkingContext { \
  public: \
    static std::unique_ptr<ELFLinkingContext> create(llvm::Triple); \
  };

// FIXME: #include "llvm/Config/Targets.def"
LLVM_TARGET(AArch64)
LLVM_TARGET(ARM)
LLVM_TARGET(Hexagon)
LLVM_TARGET(Mips)
LLVM_TARGET(X86)
LLVM_TARGET(X86_64)

#undef LLVM_TARGET

} // end namespace elf
} // end namespace lld

#endif
