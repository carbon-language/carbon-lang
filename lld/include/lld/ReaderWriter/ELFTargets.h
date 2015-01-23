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
    TargetName##LinkingContext(llvm::Triple); \
  };
#include "llvm/Config/Targets.def"

// X86 => X86,X86_64
class X86_64LinkingContext final : public ELFLinkingContext {
public:
  X86_64LinkingContext(llvm::Triple);
};

// PowerPC => PPC
class PPCLinkingContext final : public ELFLinkingContext {
public:
  PPCLinkingContext(llvm::Triple);
};

} // end namespace elf
} // end namespace lld

#endif
